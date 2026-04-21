// Integration test: .profdata round-trip through the filesystem.
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0
//
// Reference: designs/2026-04-18-pgo-workflow.md
// Issue: #396
//
// Scenario:
//   1. Build a MachFunction with a small CFG.
//   2. Run the PGO counter-injection pass; collect the CounterMap.
//   3. Simulate a canary run by producing a counter_values vector.
//   4. Call build_profdata_from_counters to get a ProfData.
//   5. Write to a temp file with write_to_path.
//   6. Read back with read_from_path.
//   7. Verify per-block hits survive the round trip.
//   8. Verify freshness check accepts the original hash and rejects a
//      perturbed hash.

use std::env;
use std::fs;
use std::path::PathBuf;

use llvm2_ir::{AArch64Opcode, MachFunction, MachInst, MachOperand, Signature};

use llvm2_opt::pgo::{
    self, CounterMap, ProfData, ProfDataError, build_profdata_from_counters, enforce_fresh,
    inject_block_counters, read_from_path, write_to_path,
};

/// Build a three-block function:
///
/// ```text
///   bb0 (entry) -> bb1 -> bb2
/// ```
fn make_three_block_function(name: &str) -> MachFunction {
    let mut f = MachFunction::new(name.to_string(), Signature::new(vec![], vec![]));
    let bb0 = f.entry;
    let bb1 = f.create_block();
    let bb2 = f.create_block();

    let br0 = f.push_inst(MachInst::new(
        AArch64Opcode::B,
        vec![MachOperand::Block(bb1)],
    ));
    f.append_inst(bb0, br0);

    let br1 = f.push_inst(MachInst::new(
        AArch64Opcode::B,
        vec![MachOperand::Block(bb2)],
    ));
    f.append_inst(bb1, br1);

    let ret = f.push_inst(MachInst::new(AArch64Opcode::Ret, vec![]));
    f.append_inst(bb2, ret);

    f.add_edge(bb0, bb1);
    f.add_edge(bb1, bb2);
    f
}

fn tmp_profdata_path(label: &str) -> PathBuf {
    env::temp_dir().join(format!(
        "llvm2_pgo_{}_{}.profdata",
        label,
        std::process::id()
    ))
}

#[test]
fn round_trip_single_function() {
    let mut f = make_three_block_function("bfs_step");
    let map = inject_block_counters(&mut f);
    assert_eq!(map.len(), 3);

    // Simulated counter array: bb0 hot, bb1 warm, bb2 cold.
    let counters = vec![10_000_u64, 9_750, 250];
    let module_hash: u128 = 0xdead_beef_cafe_babe_0123_4567_89ab_cdef;

    let original = build_profdata_from_counters(module_hash, &map, &counters);
    let path = tmp_profdata_path("single");
    write_to_path(&original, &path).unwrap();

    let loaded = read_from_path(&path).unwrap();
    assert_eq!(loaded, original, "round trip must preserve profile data");

    // Hit counts must be exactly preserved.
    let fp = loaded.function("bfs_step").unwrap();
    assert_eq!(fp.call_count, 10_000);
    assert_eq!(fp.block_hits(0), 10_000);
    assert_eq!(fp.block_hits(1), 9_750);
    assert_eq!(fp.block_hits(2), 250);

    // Hash freshness: matching hash accepted, mismatched hash rejected.
    enforce_fresh(&loaded, module_hash).unwrap();
    match enforce_fresh(&loaded, module_hash.wrapping_add(1)) {
        Err(ProfDataError::StaleHash { .. }) => {}
        other => panic!("expected StaleHash, got {:?}", other),
    }

    let _ = fs::remove_file(&path);
}

#[test]
fn round_trip_multi_function() {
    // Inject into two functions, simulate counters, round-trip.
    let mut f1 = make_three_block_function("hot_fn");
    let mut f2 = make_three_block_function("cold_fn");
    let mut map = CounterMap::new();
    map.extend(inject_block_counters(&mut f1));
    map.extend(inject_block_counters(&mut f2));
    assert_eq!(map.len(), 6);

    // hot_fn: steady; cold_fn: never entered.
    let counters = vec![500_000, 499_999, 498_000, 0, 0, 0];
    let module_hash: u128 = 0xfeed_face_0000_1111_2222_3333_4444_5555;

    let original = build_profdata_from_counters(module_hash, &map, &counters);
    let path = tmp_profdata_path("multi");
    write_to_path(&original, &path).unwrap();

    let loaded = read_from_path(&path).unwrap();
    assert_eq!(loaded, original);

    let hot = loaded.function("hot_fn").unwrap();
    assert_eq!(hot.call_count, 500_000);
    assert_eq!(hot.block_hits(0), 500_000);
    assert_eq!(hot.block_hits(1), 499_999);
    assert_eq!(hot.block_hits(2), 498_000);

    let cold = loaded.function("cold_fn").unwrap();
    assert_eq!(cold.call_count, 0);
    assert_eq!(cold.block_hits(0), 0);
    assert_eq!(cold.block_hits(1), 0);
    assert_eq!(cold.block_hits(2), 0);

    let _ = fs::remove_file(&path);
}

#[test]
fn reader_rejects_corrupt_file() {
    // Write arbitrary garbage and confirm the reader errors rather than
    // panicking.
    let path = tmp_profdata_path("corrupt");
    fs::write(&path, b"not a valid profdata").unwrap();
    let err = read_from_path(&path).unwrap_err();
    // serde_json fails to parse this as a JSON object — decoded error
    // kind must be Serde.
    assert!(matches!(err, ProfDataError::Serde(_)));
    let _ = fs::remove_file(&path);
}

#[test]
fn reader_rejects_bad_magic_file() {
    // Valid JSON but wrong magic string — must report BadMagic.
    let path = tmp_profdata_path("bad_magic");
    let mut p = ProfData::new(0);
    p.magic = "NOTLLVM2".to_string();
    let bytes = pgo::encode(&p).unwrap();
    fs::write(&path, bytes).unwrap();
    let err = read_from_path(&path).unwrap_err();
    assert!(matches!(err, ProfDataError::BadMagic { .. }));
    let _ = fs::remove_file(&path);
}
