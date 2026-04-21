use std::sync::Arc;

use llvm2_ir::{
    AArch64Opcode, MachFunction, MachInst, MachOperand, RegClass, Signature, Type, VReg,
};
use llvm2_opt::{CacheBackend, InMemoryCache, MachinePass};
use llvm2_verify::{CegisSuperoptConfig, CegisSuperoptPass};

fn make_config(cache: Option<Arc<dyn CacheBackend>>) -> CegisSuperoptConfig {
    CegisSuperoptConfig {
        budget_sec: 10,
        per_query_ms: 1_000,
        target_triple: "aarch64-apple-darwin".to_string(),
        cpu: "apple-m1".to_string(),
        features: vec!["neon".to_string(), "fp-armv8".to_string()],
        opt_level: 2,
        cache,
        trace: None,
    }
}

fn layer_a_func() -> (MachFunction, VReg, VReg, VReg) {
    let mut func = MachFunction::new(
        "layer_a_mul_zero".to_string(),
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

    (func, v0, v1, v2)
}

#[test]
fn layer_a_rewrites_mul_by_movz_zero() {
    let cache: Arc<dyn CacheBackend> = Arc::new(InMemoryCache::new());
    let mut pass = CegisSuperoptPass::new(make_config(Some(cache)));
    let (mut func, _v0, _v1, v2) = layer_a_func();

    let committed = pass.run(&mut func);

    assert!(committed, "Layer A should commit the MUL-by-zero rewrite");

    let rewritten = &func.insts[2];
    assert_eq!(rewritten.opcode, AArch64Opcode::Movz);
    assert_eq!(rewritten.operands.len(), 2);
    assert_eq!(rewritten.operands[0], MachOperand::VReg(v2));
    assert_eq!(rewritten.operands[1], MachOperand::Imm(0));

    assert!(pass.stats().verified >= 1);
    assert!(pass.stats().candidates >= 1);
}

#[test]
fn layer_a_cache_hit_replays_rewrites_issue_491() {
    // Regression test for #491: prior to the fix, the Layer A cache-hit
    // path incremented `stats.verified` but never called `apply_plan` to
    // replay the cached rewrites. Duplicate compiles silently skipped the
    // rewrites (regressing output) and dashboards reported phantom
    // verification counts.
    //
    // Post-fix invariants pinned by this test:
    //   (1) Cold run: miss + put, rewrites applied, stats.verified >= 1.
    //   (2) Hot run (same cache, identical fresh function): hit, rewrites
    //       replayed via the matcher, mutated function is byte-identical
    //       to the cold-run post-state, stats.verified reflects applied
    //       rewrites this run (not phantom cached counts).
    //   (3) `run` returns `true` on hit when replay mutated the function.
    let cache: Arc<dyn CacheBackend> = Arc::new(InMemoryCache::new());
    let cfg = make_config(Some(cache.clone()));

    // --- Cold run -----------------------------------------------------
    let mut pass_cold = CegisSuperoptPass::new(cfg.clone());
    let (mut func_cold, _v0c, _v1c, v2c) = layer_a_func();
    let committed_cold = pass_cold.run(&mut func_cold);
    assert!(committed_cold, "cold run must mutate");
    assert_eq!(pass_cold.stats().cache_misses, 1);
    assert_eq!(pass_cold.stats().cache_hits, 0);
    assert_eq!(pass_cold.stats().cache_puts, 1);
    let verified_cold = pass_cold.stats().verified;
    assert!(
        verified_cold >= 1,
        "cold run must report >= 1 verification"
    );
    // Cold post-state: insts[2] is Movz v2, #0.
    let cold_post = func_cold.insts[2].clone();
    assert_eq!(cold_post.opcode, AArch64Opcode::Movz);
    assert_eq!(cold_post.operands[0], MachOperand::VReg(v2c));
    assert_eq!(cold_post.operands[1], MachOperand::Imm(0));

    // --- Hot run (same cache, fresh identical function) ---------------
    let mut pass_hot = CegisSuperoptPass::new(cfg);
    let (mut func_hot, _v0h, _v1h, v2h) = layer_a_func();
    // Pre-run state: the MulRR at insts[2] is still intact.
    assert_eq!(func_hot.insts[2].opcode, AArch64Opcode::MulRR);

    let committed_hot = pass_hot.run(&mut func_hot);
    assert!(
        committed_hot,
        "hot run must replay cached rewrites and report mutation (#491)"
    );
    assert_eq!(pass_hot.stats().cache_hits, 1);
    assert_eq!(pass_hot.stats().cache_misses, 0);
    assert_eq!(
        pass_hot.stats().cache_puts, 0,
        "hot run must not re-put on a hit"
    );

    // Hot post-state must be byte-identical to cold post-state at insts[2].
    let hot_post = &func_hot.insts[2];
    assert_eq!(hot_post.opcode, cold_post.opcode);
    assert_eq!(hot_post.operands.len(), cold_post.operands.len());
    for (h, c) in hot_post.operands.iter().zip(cold_post.operands.iter()) {
        assert_eq!(h, c, "hot/cold operands must match byte-for-byte");
    }
    // And directly sanity-check the replayed shape:
    assert_eq!(hot_post.opcode, AArch64Opcode::Movz);
    assert_eq!(hot_post.operands[0], MachOperand::VReg(v2h));
    assert_eq!(hot_post.operands[1], MachOperand::Imm(0));

    // Stats: `verified` on the hot run must count rewrites actually
    // applied this run, matching the cold run's applied count. Before
    // #491 was fixed, the hot run reported `verified == 0` (no apply)
    // OR, worse, double-counted the cached entry — both wrong.
    assert_eq!(
        pass_hot.stats().verified,
        verified_cold,
        "hot-run `verified` must reflect applied rewrites, not phantom cache counts"
    );
    assert!(
        pass_hot.stats().candidates >= pass_hot.stats().verified,
        "candidates must be at least the number of successful replays"
    );
}
