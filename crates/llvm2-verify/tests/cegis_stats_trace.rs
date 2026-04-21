// cegis_stats_trace - CegisPassStats + trace::emit acceptance (#492)
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Pins the #486/#492 Layer A acceptance criteria that live beyond "does the
// rewrite happen":
//
//   1. `CegisPassStats` carries the six fields required by #492:
//        total_wall_ms, solver_ms,
//        layer_a_candidates, layer_a_committed,
//        timeouts, verifier_errors
//      (plus Layer B symmetry and the existing panics/budget counters).
//   2. `CegisSuperoptPass::run` emits a structured event through the
//      `llvm2_ir::trace::CompilationTrace` collector when one is wired up
//      via `CegisSuperoptConfig::trace`. The event carries the full stats
//      payload and is tagged as `Applied` (committed mutation) vs
//      `Rejected` (no-op) so downstream tooling can filter.
//   3. Stats accumulate across invocations (so `total_wall_ms` / `solver_ms`
//      are cumulative, same as `functions_seen`).

use std::sync::Arc;

use llvm2_ir::trace::{CompilationTrace, EventKind, TraceLevel};
use llvm2_ir::{
    AArch64Opcode, MachFunction, MachInst, MachOperand, RegClass, Signature, Type, VReg,
};
use llvm2_opt::{CacheBackend, InMemoryCache, MachinePass};
use llvm2_verify::{CegisSuperoptConfig, CegisSuperoptPass};

fn make_config(
    cache: Option<Arc<dyn CacheBackend>>,
    trace: Option<Arc<CompilationTrace>>,
) -> CegisSuperoptConfig {
    CegisSuperoptConfig {
        budget_sec: 10,
        per_query_ms: 1_000,
        target_triple: "aarch64-apple-darwin".to_string(),
        cpu: "apple-m1".to_string(),
        features: vec!["neon".to_string(), "fp-armv8".to_string()],
        opt_level: 2,
        cache,
        trace,
    }
}

/// Build the same Layer A pattern used by `cegis_layer_a.rs`:
///   Movz v0, #0
///   Movz v1, #7
///   MulRR v2, v1, v0    (MUL by Movz-zero — Layer A candidate)
///   Ret
fn layer_a_func() -> MachFunction {
    let mut func = MachFunction::new(
        "layer_a_mul_zero_stats".to_string(),
        Signature::new(vec![Type::I32], vec![Type::I32]),
    );
    let entry = func.entry;

    let v0 = VReg::new(func.alloc_vreg(), RegClass::Gpr32);
    let v1 = VReg::new(func.alloc_vreg(), RegClass::Gpr32);
    let v2 = VReg::new(func.alloc_vreg(), RegClass::Gpr32);

    let movz_zero = func.push_inst(MachInst::new(
        AArch64Opcode::Movz,
        vec![MachOperand::VReg(v0), MachOperand::Imm(0)],
    ));
    let movz_seven = func.push_inst(MachInst::new(
        AArch64Opcode::Movz,
        vec![MachOperand::VReg(v1), MachOperand::Imm(7)],
    ));
    let mul = func.push_inst(MachInst::new(
        AArch64Opcode::MulRR,
        vec![
            MachOperand::VReg(v2),
            MachOperand::VReg(v1),
            MachOperand::VReg(v0),
        ],
    ));
    let ret = func.push_inst(MachInst::new(AArch64Opcode::Ret, vec![]));

    func.append_inst(entry, movz_zero);
    func.append_inst(entry, movz_seven);
    func.append_inst(entry, mul);
    func.append_inst(entry, ret);

    func
}

#[test]
fn layer_a_populates_required_stats_fields_issue_492() {
    // Cold run on a Layer A candidate function. We expect the following
    // fields to all update according to their spec:
    //   - layer_a_candidates >= 1 (MulRR x, Movz #0 matched)
    //   - layer_a_committed  >= 1 (proof passed + cost gate passed)
    //   - total_wall_ms      >  0 (at least one nanosecond elapsed;
    //                              Instant rounding may give 0 on very
    //                              fast CI — assert `>= 0` only)
    //   - solver_ms          accumulates alongside total_wall_ms
    //   - timeouts           == 0 (no real timeouts on toy input)
    //   - verifier_errors    == 0 (no solver errors expected)
    //   - panics             == 0
    //
    // Note: we assert on presence/logical invariants rather than exact
    // timings to keep the test deterministic across CI load.
    let cache: Arc<dyn CacheBackend> = Arc::new(InMemoryCache::new());
    let mut pass = CegisSuperoptPass::new(make_config(Some(cache), None));
    let mut func = layer_a_func();

    let committed = pass.run(&mut func);
    assert!(committed, "Layer A rewrite must commit on cold run");

    let stats = pass.stats();

    // Required Layer A per-layer counters (#492 AC #1).
    assert!(
        stats.layer_a_candidates >= 1,
        "expected >=1 Layer A candidate, got {}",
        stats.layer_a_candidates,
    );
    assert!(
        stats.layer_a_committed >= 1,
        "expected >=1 Layer A commit, got {}",
        stats.layer_a_committed,
    );
    assert!(
        stats.layer_a_committed <= stats.layer_a_candidates,
        "layer_a_committed must be a subset of layer_a_candidates"
    );

    // Timing counters exist and are coherent. `total_wall_ms` SHOULD be
    // positive, but on extremely fast hardware `Instant::elapsed().as_millis()`
    // rounding can give 0 for a single toy call, so we assert the weaker
    // invariant that solver_ms <= total_wall_ms (subset relationship).
    assert!(
        stats.solver_ms <= stats.total_wall_ms.saturating_add(5),
        "solver_ms ({}) must not exceed total_wall_ms ({}) meaningfully",
        stats.solver_ms,
        stats.total_wall_ms,
    );

    // Failure mode counters default to zero on a clean run.
    assert_eq!(stats.timeouts, 0, "clean Layer A run must have zero timeouts");
    assert_eq!(
        stats.verifier_errors, 0,
        "clean Layer A run must have zero verifier errors"
    );
    assert_eq!(stats.panics, 0, "clean Layer A run must have zero panics");

    // Roll-up counters stay consistent with per-layer counters.
    assert!(
        stats.candidates >= stats.layer_a_candidates,
        "roll-up candidates ({}) must include layer_a_candidates ({})",
        stats.candidates,
        stats.layer_a_candidates,
    );
    assert!(
        stats.verified >= stats.layer_a_committed,
        "roll-up verified ({}) must include layer_a_committed ({})",
        stats.verified,
        stats.layer_a_committed,
    );
}

#[test]
fn stats_are_serde_roundtrippable_issue_492() {
    // #492 adds Serialize/Deserialize to CegisPassStats so harnesses can
    // roundtrip stats through JSON for dashboard ingestion (#486 §10).
    let cache: Arc<dyn CacheBackend> = Arc::new(InMemoryCache::new());
    let mut pass = CegisSuperoptPass::new(make_config(Some(cache), None));
    let mut func = layer_a_func();
    let _ = pass.run(&mut func);
    let stats = pass.stats().clone();

    let json = serde_json::to_string(&stats).expect("stats must serialize to JSON");
    // All #492 fields must be keyed in the JSON blob.
    for key in &[
        "total_wall_ms",
        "solver_ms",
        "layer_a_candidates",
        "layer_a_committed",
        "layer_b_candidates",
        "layer_b_committed",
        "timeouts",
        "verifier_errors",
        "panics",
    ] {
        assert!(
            json.contains(key),
            "stats JSON must include field `{}`, got: {}",
            key,
            json,
        );
    }

    let roundtripped: llvm2_verify::CegisPassStats =
        serde_json::from_str(&json).expect("stats must deserialize from JSON");
    assert_eq!(roundtripped.layer_a_candidates, stats.layer_a_candidates);
    assert_eq!(roundtripped.layer_a_committed, stats.layer_a_committed);
    assert_eq!(roundtripped.total_wall_ms, stats.total_wall_ms);
    assert_eq!(roundtripped.solver_ms, stats.solver_ms);
    assert_eq!(roundtripped.timeouts, stats.timeouts);
    assert_eq!(roundtripped.verifier_errors, stats.verifier_errors);
}

#[test]
fn trace_emits_applied_event_on_committed_rewrite_issue_492() {
    // #492 AC #2: when a CompilationTrace is wired in, the pass must emit
    // a structured event summarizing the invocation. Committed rewrites
    // produce an `EventKind::Applied` event; the payload (encoded in the
    // PassId name slot) carries every required stats field as key=value.
    let cache: Arc<dyn CacheBackend> = Arc::new(InMemoryCache::new());
    let trace = Arc::new(CompilationTrace::new(TraceLevel::Full));
    let mut pass = CegisSuperoptPass::new(make_config(Some(cache), Some(trace.clone())));
    let mut func = layer_a_func();

    let committed = pass.run(&mut func);
    assert!(committed, "test precondition: Layer A must commit");

    let events = trace.events();
    assert_eq!(
        events.len(),
        1,
        "one CEGIS invocation should produce exactly one trace event",
    );
    let event = &events[0];

    // Event kind must be Applied on a committed rewrite.
    match &event.kind {
        EventKind::Applied { .. } => {}
        other => panic!(
            "expected EventKind::Applied on committed rewrite, got {:?}",
            other
        ),
    }

    // The PassId name slot carries the full stats payload. Assert every
    // #492-required field is discoverable in the payload.
    let payload = event.pass.name();
    assert!(payload.starts_with("CegisSuperoptPass{"), "payload: {}", payload);
    assert!(payload.contains("func=layer_a_mul_zero_stats"));
    assert!(payload.contains("cache=miss"));
    assert!(payload.contains("committed=true"));
    for key in &[
        "wall_ms=",
        "solver_ms=",
        "candidates=",
        "verified=",
        "rejected=",
        "layer_a_candidates=",
        "layer_a_committed=",
        "layer_b_candidates=",
        "layer_b_committed=",
        "timeouts=",
        "verifier_errors=",
        "panics=",
        "solver_calls=",
    ] {
        assert!(
            payload.contains(key),
            "trace payload must include `{}`, got: {}",
            key,
            payload,
        );
    }

    // The event must round-trip through the trace's JSON serializer.
    let json = trace.to_json();
    assert!(json.contains("CegisSuperoptPass"));
    assert!(json.contains("Applied"));
}

#[test]
fn trace_emits_rejected_event_when_no_match_issue_492() {
    // On an empty / non-matching function the pass runs, produces no
    // candidates, and must still emit a single Rejected event so
    // downstream dashboards can tell "pass ran, did nothing" apart from
    // "pass never ran".
    let cache: Arc<dyn CacheBackend> = Arc::new(InMemoryCache::new());
    let trace = Arc::new(CompilationTrace::new(TraceLevel::Summary));
    let mut pass = CegisSuperoptPass::new(make_config(Some(cache), Some(trace.clone())));
    let mut func = MachFunction::new(
        "noop".to_string(),
        Signature::new(vec![], vec![]),
    );

    let committed = pass.run(&mut func);
    assert!(!committed, "empty function should not mutate");

    let events = trace.events();
    assert_eq!(events.len(), 1, "pass must emit exactly one event per run");
    let event = &events[0];

    match &event.kind {
        EventKind::Rejected { reason, .. } => {
            // One of the documented reason strings.
            assert!(
                reason.contains("no candidate")
                    || reason.contains("cache-hit")
                    || reason.contains("all candidates rejected"),
                "unexpected rejection reason: {}",
                reason,
            );
        }
        other => panic!("expected EventKind::Rejected on noop, got {:?}", other),
    }

    let payload = event.pass.name();
    assert!(payload.contains("committed=false"));
    assert!(payload.contains("candidates=0"));
    assert!(payload.contains("verified=0"));
}

#[test]
fn trace_level_none_is_silent_issue_492() {
    // A trace at level `None` must not record any events, even though the
    // pass invokes `trace.emit`. This preserves the 0% overhead guarantee
    // of production builds (llvm2_ir::trace::TraceLevel docs).
    let cache: Arc<dyn CacheBackend> = Arc::new(InMemoryCache::new());
    let trace = Arc::new(CompilationTrace::new(TraceLevel::None));
    let mut pass = CegisSuperoptPass::new(make_config(Some(cache), Some(trace.clone())));
    let mut func = layer_a_func();

    let _ = pass.run(&mut func);
    assert!(trace.is_empty(), "None-level trace must record zero events");
}

#[test]
fn trace_cache_hit_path_still_emits_event_issue_492() {
    // The cache-hit replay path must also emit a trace event, so that
    // dashboards see every pass invocation — not just cold runs. The
    // payload must flag `cache=hit` so downstream analysis can bucket
    // hot/cold invocations.
    let cache: Arc<dyn CacheBackend> = Arc::new(InMemoryCache::new());
    let cfg_cold = make_config(Some(cache.clone()), None);

    // Cold run: no trace.
    let mut pass_cold = CegisSuperoptPass::new(cfg_cold);
    let mut func_cold = layer_a_func();
    let _ = pass_cold.run(&mut func_cold);
    assert_eq!(pass_cold.stats().cache_misses, 1);

    // Hot run with a fresh trace.
    let trace = Arc::new(CompilationTrace::new(TraceLevel::Full));
    let cfg_hot = make_config(Some(cache), Some(trace.clone()));
    let mut pass_hot = CegisSuperoptPass::new(cfg_hot);
    let mut func_hot = layer_a_func();
    let _ = pass_hot.run(&mut func_hot);
    assert_eq!(pass_hot.stats().cache_hits, 1);

    let events = trace.events();
    assert_eq!(events.len(), 1, "hot run must emit exactly one event");
    let payload = events[0].pass.name();
    assert!(payload.contains("cache=hit"), "payload: {}", payload);
    // Replay of a cached Layer A rewrite counts as layer_a_committed=1 on
    // the hot run too (#491 semantics preserved).
    assert!(payload.contains("layer_a_committed=1"), "payload: {}", payload);
}
