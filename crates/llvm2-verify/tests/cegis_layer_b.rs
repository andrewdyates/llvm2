// cegis_layer_b - Layer B (two-instruction window fusion) integration test
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0
//
// Mirrors `cegis_layer_a.rs` for the two-instruction window fusion layer of
// the CEGIS superopt payload (issue #486, Layer B).
//
// Rule under test:
//   Movz  v,   #imm        (single-use)
//   AddRR dst, src, v     ->  AddRI dst, src, imm
//
// Acceptance:
//   - `CegisSuperoptPass::run` returns true.
//   - The AddRR at its original InstId is rewritten in place to AddRI with
//     the original destination VReg preserved (SSA invariant held).
//   - The Movz InstId is no longer scheduled in its block's `insts` list.
//   - Pass stats reflect at least one candidate and one verification.

use std::sync::Arc;

use llvm2_ir::{
    AArch64Opcode, InstId, MachFunction, MachInst, MachOperand, RegClass, Signature, Type, VReg,
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

/// Build a minimal single-block function exercising the Layer B pattern:
///
///   Movz  v_imm, #7
///   AddRR v_dst, v_src, v_imm
///   Ret
///
/// `v_imm` is defined exactly once (by the Movz) and used exactly once
/// (by the AddRR) so the single-use gate in the matcher is satisfied.
fn layer_b_func() -> (MachFunction, VReg, VReg, VReg, InstId, InstId, InstId) {
    let mut func = MachFunction::new(
        "layer_b_movz_add".to_string(),
        Signature::new(vec![Type::I32], vec![Type::I32]),
    );
    let entry = func.entry;

    let v_src = VReg::new(func.alloc_vreg(), RegClass::Gpr32);
    let v_imm = VReg::new(func.alloc_vreg(), RegClass::Gpr32);
    let v_dst = VReg::new(func.alloc_vreg(), RegClass::Gpr32);

    let movz = func.push_inst(MachInst::new(
        AArch64Opcode::Movz,
        vec![MachOperand::VReg(v_imm), MachOperand::Imm(7)],
    ));
    let add = func.push_inst(MachInst::new(
        AArch64Opcode::AddRR,
        vec![
            MachOperand::VReg(v_dst),
            MachOperand::VReg(v_src),
            MachOperand::VReg(v_imm),
        ],
    ));
    let ret = func.push_inst(MachInst::new(AArch64Opcode::Ret, vec![]));

    func.append_inst(entry, movz);
    func.append_inst(entry, add);
    func.append_inst(entry, ret);

    (func, v_src, v_imm, v_dst, movz, add, ret)
}

#[test]
fn layer_b_rewrites_movz_add_pair() {
    let cache: Arc<dyn CacheBackend> = Arc::new(InMemoryCache::new());
    let mut pass = CegisSuperoptPass::new(make_config(Some(cache)));
    let (mut func, v_src, _v_imm, v_dst, movz_id, add_id, _ret_id) = layer_b_func();

    let committed = pass.run(&mut func);
    assert!(committed, "Layer B should commit the Movz+AddRR fusion");

    // The AddRR's InstId must now hold an AddRI with the original dst VReg,
    // the original src VReg, and the Movz's immediate (#7).
    let rewritten = func.inst(add_id);
    assert_eq!(rewritten.opcode, AArch64Opcode::AddRI);
    assert_eq!(rewritten.operands.len(), 3);
    assert_eq!(rewritten.operands[0], MachOperand::VReg(v_dst));
    assert_eq!(rewritten.operands[1], MachOperand::VReg(v_src));
    assert_eq!(rewritten.operands[2], MachOperand::Imm(7));

    // The Movz must be spliced out of the block's instruction list.
    let block = func.block(func.entry);
    assert!(
        !block.insts.contains(&movz_id),
        "spliced Movz must not remain in the block schedule"
    );
    assert!(
        block.insts.contains(&add_id),
        "fused AddRI must remain scheduled at the original AddRR InstId"
    );

    // Arena entry for the Movz may still exist (orphaned); verify the arena
    // is sound regardless: the fused AddRI at `add_id` is what regalloc and
    // codegen will consume.
    assert_eq!(func.inst(add_id).opcode, AArch64Opcode::AddRI);

    // Stats reflect at least one candidate and one verification.
    assert!(pass.stats().candidates >= 1);
    assert!(pass.stats().verified >= 1);
}

#[test]
fn layer_b_does_not_fuse_when_movz_has_extra_use() {
    // Pattern:
    //   Movz  v_imm, #7
    //   AddRR v_dst1, v_src, v_imm
    //   AddRR v_dst2, v_src, v_imm     (second consumer of v_imm)
    //
    // The single-use gate must prevent fusion; otherwise the second AddRR
    // would dangle on a dead Movz. Layer A does not fire here either (no
    // MUL-by-zero shape), so we expect `committed == false`.
    let cache: Arc<dyn CacheBackend> = Arc::new(InMemoryCache::new());
    let mut pass = CegisSuperoptPass::new(make_config(Some(cache)));

    let mut func = MachFunction::new(
        "layer_b_multi_use".to_string(),
        Signature::new(vec![Type::I32], vec![Type::I32]),
    );
    let entry = func.entry;

    let v_src = VReg::new(func.alloc_vreg(), RegClass::Gpr32);
    let v_imm = VReg::new(func.alloc_vreg(), RegClass::Gpr32);
    let v_dst1 = VReg::new(func.alloc_vreg(), RegClass::Gpr32);
    let v_dst2 = VReg::new(func.alloc_vreg(), RegClass::Gpr32);

    let movz = func.push_inst(MachInst::new(
        AArch64Opcode::Movz,
        vec![MachOperand::VReg(v_imm), MachOperand::Imm(7)],
    ));
    let add1 = func.push_inst(MachInst::new(
        AArch64Opcode::AddRR,
        vec![
            MachOperand::VReg(v_dst1),
            MachOperand::VReg(v_src),
            MachOperand::VReg(v_imm),
        ],
    ));
    let add2 = func.push_inst(MachInst::new(
        AArch64Opcode::AddRR,
        vec![
            MachOperand::VReg(v_dst2),
            MachOperand::VReg(v_src),
            MachOperand::VReg(v_imm),
        ],
    ));
    let ret = func.push_inst(MachInst::new(AArch64Opcode::Ret, vec![]));

    func.append_inst(entry, movz);
    func.append_inst(entry, add1);
    func.append_inst(entry, add2);
    func.append_inst(entry, ret);

    let committed = pass.run(&mut func);
    assert!(
        !committed,
        "Layer B must not fuse when the Movz has multiple consumers"
    );

    // The Movz must still be scheduled — nothing was spliced.
    let block = func.block(entry);
    assert!(block.insts.contains(&movz));
    // Both AddRRs must remain unchanged.
    assert_eq!(func.inst(add1).opcode, AArch64Opcode::AddRR);
    assert_eq!(func.inst(add2).opcode, AArch64Opcode::AddRR);

    // No successful verifications.
    assert_eq!(pass.stats().verified, 0);
}

#[test]
fn layer_b_cache_replay_on_second_run_applies_rewrite() {
    // Issue #491: a bit-identical second pass must replay the cached
    // rewrite so the output is identical to the cold-path run. Prior to
    // #491 the hit path was a silent no-op, causing duplicate compiles
    // to regress their output. This test pins the corrected behavior.
    let cache: Arc<dyn CacheBackend> = Arc::new(InMemoryCache::new());
    let cfg = make_config(Some(cache.clone()));

    // First run: miss → verify → put.
    let mut pass1 = CegisSuperoptPass::new(cfg.clone());
    let (mut func1, v_src1, _, v_dst1, movz_id1, add_id1, _) = layer_b_func();
    let committed1 = pass1.run(&mut func1);
    assert!(committed1);
    assert_eq!(pass1.stats().cache_misses, 1);
    assert_eq!(pass1.stats().cache_hits, 0);
    assert_eq!(pass1.stats().cache_puts, 1);
    let verified_cold = pass1.stats().verified;
    assert!(verified_cold >= 1);
    // Cold-path post-state: AddRI at add_id1, Movz spliced out.
    assert_eq!(func1.inst(add_id1).opcode, AArch64Opcode::AddRI);
    assert!(!func1.block(func1.entry).insts.contains(&movz_id1));

    // Second run on an identical (fresh) function: hit with replay.
    let mut pass2 = CegisSuperoptPass::new(cfg);
    let (mut func2, v_src2, _, v_dst2, movz_id2, add_id2, _) = layer_b_func();
    let committed2 = pass2.run(&mut func2);
    assert!(committed2, "hit path must replay rewrites and report mutation");
    assert_eq!(pass2.stats().cache_hits, 1);
    assert_eq!(pass2.stats().cache_misses, 0);

    // Post-replay state must mirror the cold-path state byte-for-byte.
    let add_hot = func2.inst(add_id2);
    let add_cold = func1.inst(add_id1);
    assert_eq!(add_hot.opcode, AArch64Opcode::AddRI);
    assert_eq!(add_hot.opcode, add_cold.opcode);
    assert_eq!(add_hot.operands.len(), add_cold.operands.len());
    assert_eq!(add_hot.operands[0], MachOperand::VReg(v_dst2));
    assert_eq!(add_hot.operands[1], MachOperand::VReg(v_src2));
    assert_eq!(add_hot.operands[2], MachOperand::Imm(7));
    assert!(!func2.block(func2.entry).insts.contains(&movz_id2));
    // Parallel structure between hot and cold:
    assert_eq!(v_src1.id, v_src2.id);
    assert_eq!(v_dst1.id, v_dst2.id);

    // Stats on the hot run must reflect rewrites actually applied this run
    // — NOT the phantom cached `entry.verified` count. Applied rewrites
    // should match what the cold run produced (same function, same
    // deterministic matcher input).
    assert_eq!(
        pass2.stats().verified,
        verified_cold,
        "hot-run `verified` must equal rewrites actually applied, matching cold run"
    );
}
