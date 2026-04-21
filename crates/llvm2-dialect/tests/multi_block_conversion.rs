// llvm2-dialect - Multi-block conversion regression tests (F2)
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! Regression tests for issue #392 / auditor finding F2.
//!
//! Prior to this fix, [`ConversionDriver::run_on_function`] allocated a single
//! destination block and stuffed every source op into it regardless of which
//! source block it originated in. That silently collapsed multi-block
//! functions and misplaced control-flow terminators mid-function. These tests
//! build source modules with 2+ blocks, run conversion, and verify:
//!
//! * block count is preserved,
//! * ops end up in the destination block matching their source block,
//! * `Attribute::Block` references (CFG edges) are remapped,
//! * block parameters survive with their `ValueId`s re-allocated and types
//!   preserved.

use llvm2_ir::Type;

use llvm2_dialect::conversion::{
    ConversionDriver, ConversionError, ConversionPattern, Rewriter,
};
use llvm2_dialect::dialects::conversions::register_all;
use llvm2_dialect::dialects::{tmir, verif};
use llvm2_dialect::id::{BlockId, DialectOpId};
use llvm2_dialect::module::{DialectFunction, DialectModule};
use llvm2_dialect::op::{Attribute, DialectOp};
use llvm2_dialect::registry::DialectRegistry;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a three-block source function:
///
/// ```text
/// bb0(p: i64):
///   %t0 = tmir.const {value = 1}
///   tmir.add %p, %t0  // lands in bb0
///   // "control-flow" ops are encoded via Attribute::Block(target) so the
///   // remapping logic gets exercised even though the PoC dialects do not
///   // ship real branch ops yet.
///   verif.fingerprint_batch_stub(%p, %p) {goto = bb1}
/// bb1(q: i64):
///   %t1 = tmir.const {value = 2}
///   verif.fingerprint_batch_stub(%q, %q) {goto = bb2}
/// bb2:
///   tmir.ret
/// ```
///
/// The `goto` attribute is fake syntax — it is just an `Attribute::Block`
/// used so the block-remap path is hit by the conversion driver.
fn build_three_block_source() -> (
    DialectModule,
    llvm2_dialect::id::DialectId,
    llvm2_dialect::id::DialectId,
    llvm2_dialect::id::DialectId,
    Vec<BlockId>,
) {
    let mut registry = DialectRegistry::new();
    let (verif_id, tmir_id, _machir_id) = register_all(&mut registry);

    let mut f = DialectFunction::new("multi_bb", vec![Type::I64], vec![Type::I64]);
    let bb0 = f.entry_block().unwrap();
    let p = f.params[0].0;

    // bb1 carries a block parameter %q: i64.
    let (bb1, bb1_params) = f.new_block_with_params(vec![Type::I64]);
    let q = bb1_params[0];

    let bb2 = f.new_block();

    // bb0 body.
    let t0 = f.alloc_value();
    f.append_op(
        bb0,
        DialectOpId::new(tmir_id, tmir::TMIR_CONST),
        vec![(t0, Type::I64)],
        vec![],
        vec![("value".to_string(), Attribute::U64(1))],
        None,
    );
    let added = f.alloc_value();
    f.append_op(
        bb0,
        DialectOpId::new(tmir_id, tmir::TMIR_ADD),
        vec![(added, Type::I64)],
        vec![p, t0],
        vec![],
        None,
    );
    // Attribute::Block(bb1) — tests CFG remap through the block_map.
    let fp_a = f.alloc_value();
    f.append_op(
        bb0,
        DialectOpId::new(verif_id, verif::FINGERPRINT_BATCH_STUB),
        vec![(fp_a, Type::I64)],
        vec![p, p],
        vec![("goto".to_string(), Attribute::Block(bb1))],
        None,
    );

    // bb1 body (must use %q so block-param wiring is observable).
    let t1 = f.alloc_value();
    f.append_op(
        bb1,
        DialectOpId::new(tmir_id, tmir::TMIR_CONST),
        vec![(t1, Type::I64)],
        vec![],
        vec![("value".to_string(), Attribute::U64(2))],
        None,
    );
    let fp_b = f.alloc_value();
    f.append_op(
        bb1,
        DialectOpId::new(verif_id, verif::FINGERPRINT_BATCH_STUB),
        vec![(fp_b, Type::I64)],
        vec![q, q],
        vec![("goto".to_string(), Attribute::Block(bb2))],
        None,
    );

    // bb2: terminator.
    f.append_op(
        bb2,
        DialectOpId::new(tmir_id, tmir::TMIR_RET),
        vec![],
        vec![],
        vec![],
        None,
    );

    let mut module = DialectModule::new("multi_bb_mod", registry);
    module.push_function(f);
    (module, verif_id, tmir_id, _machir_id, vec![bb0, bb1, bb2])
}

// ---------------------------------------------------------------------------
// Test 1: empty driver preserves block count and edges (copy-through path).
//
// Without the F2 fix, this test would fail because every op would land in the
// single destination entry block and bb1/bb2 would be empty.
// ---------------------------------------------------------------------------

#[test]
fn empty_driver_preserves_multi_block_structure() {
    let (mut module, _verif_id, _tmir_id, _machir_id, src_blocks) =
        build_three_block_source();

    // Record expected per-block op counts up front.
    let src_fn = &module.functions[0];
    let src_op_counts: Vec<usize> =
        src_fn.blocks.iter().map(|b| b.ops.len()).collect();
    assert_eq!(src_op_counts, vec![3, 2, 1]);

    // Expected CFG edges (block -> list of successor-BlockId attrs).
    let src_edges: Vec<Vec<BlockId>> = src_fn
        .blocks
        .iter()
        .map(|b| {
            b.ops
                .iter()
                .flat_map(|op_id| src_fn.ops[op_id.0 as usize].block_refs())
                .collect()
        })
        .collect();
    assert_eq!(src_edges, vec![vec![src_blocks[1]], vec![src_blocks[2]], vec![]]);

    // Run an empty driver — every op takes the copy-through path.
    let driver = ConversionDriver::new();
    driver.run(&mut module).expect("empty driver succeeds");

    let dst = &module.functions[0];

    // Block count preserved one-for-one.
    assert_eq!(dst.blocks.len(), 3, "block count must be preserved");

    // Per-block op counts preserved.
    let dst_counts: Vec<usize> = dst.blocks.iter().map(|b| b.ops.len()).collect();
    assert_eq!(
        dst_counts, src_op_counts,
        "per-block op counts must match source"
    );

    // Block parameters: bb0 has none (function params live in
    // `DialectFunction::params`), bb1 has exactly one i64 param, bb2 none.
    assert_eq!(dst.blocks[0].params.len(), 0);
    assert_eq!(dst.blocks[1].params.len(), 1);
    assert_eq!(dst.blocks[1].params[0].1, Type::I64);
    assert_eq!(dst.blocks[2].params.len(), 0);

    // CFG edges remapped through block_map — expect bb0 -> dst.blocks[1].id,
    // bb1 -> dst.blocks[2].id, bb2 -> nothing.
    let dst_edges: Vec<Vec<BlockId>> = dst
        .blocks
        .iter()
        .map(|b| {
            b.ops
                .iter()
                .flat_map(|op_id| dst.ops[op_id.0 as usize].block_refs())
                .collect()
        })
        .collect();
    assert_eq!(
        dst_edges,
        vec![vec![dst.blocks[1].id], vec![dst.blocks[2].id], vec![]],
        "CFG edges must point to destination-space BlockIds"
    );

    // Terminator stays in bb2 (not mid-function, not in bb0). This is the
    // precise regression F2 described.
    let tmir_ret_name = module
        .resolve(dst.ops[dst.blocks[2].ops[0].0 as usize].op)
        .map(|d| d.name)
        .unwrap();
    assert_eq!(tmir_ret_name, "tmir.ret");
}

// ---------------------------------------------------------------------------
// Test 2: pattern-based conversion preserves block structure too.
//
// Register a pattern for `verif.fingerprint_batch_stub` that replaces it with
// a single `tmir.xor` op and verify the resulting function still has 3
// blocks, terminator still trails, and block params re-map onto the new
// ValueIds so block 1's ops consume the destination %q.
// ---------------------------------------------------------------------------

struct VerifStubToTmirXor {
    verif_id: llvm2_dialect::id::DialectId,
    tmir_id: llvm2_dialect::id::DialectId,
}

impl ConversionPattern for VerifStubToTmirXor {
    fn source_op(&self) -> DialectOpId {
        DialectOpId::new(self.verif_id, verif::FINGERPRINT_BATCH_STUB)
    }

    fn rewrite(
        &self,
        op: &DialectOp,
        rewriter: &mut Rewriter<'_>,
    ) -> Result<(), ConversionError> {
        assert_eq!(op.operands.len(), 2);
        let a = op.operands[0];
        let b = op.operands[1];
        let (src_result, ty) = op.results[0].clone();
        let dst = rewriter.alloc_value();
        // Preserve Attribute::Block("goto") — the driver has already remapped
        // it into destination-block space for us, so we just clone it across.
        let attrs: Vec<(String, Attribute)> = op.attrs.clone();
        rewriter.emit(
            DialectOpId::new(self.tmir_id, tmir::TMIR_XOR),
            vec![(dst, ty)],
            vec![a, b],
            attrs,
            op.source,
        );
        rewriter.bind_result(src_result, dst);
        Ok(())
    }
}

#[test]
fn pattern_conversion_preserves_multi_block_structure() {
    let (mut module, verif_id, tmir_id, _machir_id, _src_blocks) =
        build_three_block_source();

    let mut driver = ConversionDriver::new();
    driver.register(Box::new(VerifStubToTmirXor { verif_id, tmir_id }));
    driver.run(&mut module).expect("verif stub conversion succeeds");

    let dst = &module.functions[0];

    // 3 blocks preserved.
    assert_eq!(dst.blocks.len(), 3);
    let dst_counts: Vec<usize> = dst.blocks.iter().map(|b| b.ops.len()).collect();
    // bb0 still has {tmir.const, tmir.add, tmir.xor(new)} = 3 ops.
    // bb1 still has {tmir.const, tmir.xor(new)} = 2 ops.
    // bb2 still has {tmir.ret} = 1 op.
    assert_eq!(dst_counts, vec![3, 2, 1]);

    // Block params preserved on bb1.
    assert_eq!(dst.blocks[1].params.len(), 1);
    assert_eq!(dst.blocks[1].params[0].1, Type::I64);

    // The bb1 tmir.xor should consume the destination-space ValueId of %q
    // (the newly-allocated block param), not a stale source ValueId.
    let bb1_q_dst = dst.blocks[1].params[0].0;
    let bb1_ops: Vec<&DialectOp> = dst.blocks[1]
        .ops
        .iter()
        .map(|op_id| &dst.ops[op_id.0 as usize])
        .collect();
    let xor_op = bb1_ops
        .iter()
        .find(|o| {
            module
                .resolve(o.op)
                .map(|d| d.name == "tmir.xor")
                .unwrap_or(false)
        })
        .expect("bb1 has a tmir.xor");
    assert_eq!(
        xor_op.operands,
        vec![bb1_q_dst, bb1_q_dst],
        "bb1 xor must reference the destination block param %q"
    );

    // The xor op in bb0 carries a goto-Block attr that still points at a
    // valid destination block (bb1).
    let bb0_xor = dst.blocks[0]
        .ops
        .iter()
        .map(|op_id| &dst.ops[op_id.0 as usize])
        .find(|o| {
            module
                .resolve(o.op)
                .map(|d| d.name == "tmir.xor")
                .unwrap_or(false)
        })
        .expect("bb0 has a tmir.xor");
    let edges = bb0_xor.block_refs();
    assert_eq!(edges, vec![dst.blocks[1].id]);

    // Terminator still lives in bb2 (not promoted up to bb0). If F2 weren't
    // fixed, this assertion would find tmir.ret in bb0, alongside non-
    // terminator ops — exactly the "terminators mid-function" bug.
    let bb2_ret_name = module
        .resolve(dst.ops[dst.blocks[2].ops[0].0 as usize].op)
        .map(|d| d.name)
        .unwrap();
    assert_eq!(bb2_ret_name, "tmir.ret");

    // And bb0/bb1 should *not* contain a terminator.
    for block_idx in [0, 1] {
        for op_id in &dst.blocks[block_idx].ops {
            let op = &dst.ops[op_id.0 as usize];
            let def = module.resolve(op.op).expect("resolved op");
            assert!(
                !def.capabilities.is_terminator(),
                "block {} contains an unexpected terminator op {}",
                block_idx,
                def.name
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Test 3: explicit pre-fix regression — a two-block source with a terminator
// in the *last* block would have collapsed to a single block under the old
// code, so the non-terminator op in bb0 would have appeared *after* the
// tmir.ret. This assertion directly pins that failure mode.
// ---------------------------------------------------------------------------

#[test]
fn terminator_in_last_block_does_not_bubble_up() {
    let mut registry = DialectRegistry::new();
    let (_verif_id, tmir_id, _machir_id) = register_all(&mut registry);

    let mut f = DialectFunction::new("two_bb", vec![Type::I64], vec![Type::I64]);
    let bb0 = f.entry_block().unwrap();
    let p = f.params[0].0;

    let bb1 = f.new_block();

    // bb0: a single non-terminator op.
    let t0 = f.alloc_value();
    f.append_op(
        bb0,
        DialectOpId::new(tmir_id, tmir::TMIR_ADD),
        vec![(t0, Type::I64)],
        vec![p, p],
        vec![],
        None,
    );

    // bb1: the terminator, consuming bb0's result. This creates a cross-block
    // SSA edge which the old code could only satisfy by collapsing blocks.
    f.append_op(
        bb1,
        DialectOpId::new(tmir_id, tmir::TMIR_RET),
        vec![],
        vec![t0],
        vec![],
        None,
    );

    let mut module = DialectModule::new("two_bb_mod", registry);
    module.push_function(f);

    let driver = ConversionDriver::new();
    driver.run(&mut module).expect("conversion runs");

    let dst = &module.functions[0];
    assert_eq!(dst.blocks.len(), 2, "must preserve both source blocks");
    assert_eq!(
        dst.blocks[0].ops.len(),
        1,
        "bb0 must keep exactly its non-terminator op"
    );
    assert_eq!(
        dst.blocks[1].ops.len(),
        1,
        "bb1 must keep exactly the terminator"
    );

    // The terminator sits physically in bb1, after bb0.
    let last_op = &dst.ops[dst.blocks[1].ops[0].0 as usize];
    let last_def = module.resolve(last_op.op).unwrap();
    assert!(
        last_def.capabilities.is_terminator(),
        "bb1's op must be the terminator, got {}",
        last_def.name
    );

    // And the terminator's operand was remapped to the destination ValueId
    // produced by bb0's tmir.add (not left as the source ValueId).
    let bb0_add = &dst.ops[dst.blocks[0].ops[0].0 as usize];
    let bb0_add_result = bb0_add.results[0].0;
    assert_eq!(last_op.operands, vec![bb0_add_result]);
}

// ---------------------------------------------------------------------------
// Test 4: the block_map is complete — every source block must have exactly
// one destination block.
// ---------------------------------------------------------------------------

#[test]
fn block_map_is_one_to_one() {
    let (mut module, _verif_id, _tmir_id, _machir_id, _src_blocks) =
        build_three_block_source();

    let driver = ConversionDriver::new();
    driver.run(&mut module).expect("conversion runs");

    let src_len = 3;
    let dst_len = module.functions[0].blocks.len();
    assert_eq!(
        src_len, dst_len,
        "block count must match 1:1 between source and destination"
    );

    // Block ids are distinct in the destination too.
    let mut ids: Vec<u32> = module.functions[0]
        .blocks
        .iter()
        .map(|b| b.id.0)
        .collect();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(ids.len(), dst_len, "destination block ids must be unique");
}
