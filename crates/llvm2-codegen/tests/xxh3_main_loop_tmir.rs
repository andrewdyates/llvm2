// llvm2-codegen/tests/xxh3_main_loop_tmir.rs - xxh3 main-loop primitive as tMIR
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0
//
// Implements the xxh3 bulk-hashing main loop primitive as tMIR and verifies
// that LLVM2's O2 pipeline compiles it to correct machine code. This is the
// hot path for tla2's compiled state fingerprinting (issue #343) — one xxh3
// round per 8-byte chunk of a BFS state label.
//
// Operations used: 64-bit multiply, rotate (as shift+or), XOR, pointer load,
// counted loop with block-parameter-threaded accumulator.
//
// Part of #343 - Inline xxh3 hash as tMIR for compiled fingerprinting

use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use llvm2_codegen::pipeline::{OptLevel, Pipeline, PipelineConfig};
use tmir::{BinOp, ICmpOp, Ty};
use tmir_build::ModuleBuilder;

const XXH3_PRIME1: u64 = 0x9E37_79B1_85EB_CA87;
const XXH3_PRIME2: u64 = 0xC2B2_AE3D_27D4_EB4F;

const XXH3_MAIN_LOOP_DRIVER_C: &str = r#"
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

extern uint64_t xxh3_main_loop(uint64_t acc, const void* data, uint64_t nblocks);

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <acc_hex> <data_hex>\n", argv[0]); return 1; }
    uint64_t acc = strtoull(argv[1], NULL, 16);
    const char* hex = argv[2];
    size_t hexlen = strlen(hex);
    if (hexlen % 2 != 0) { fprintf(stderr, "bad hex len\n"); return 1; }
    size_t nbytes = hexlen / 2;
    uint8_t* buf = (uint8_t*)malloc(nbytes + 16);
    if (!buf) { return 1; }
    for (size_t i = 0; i < nbytes; i++) {
        unsigned int b;
        sscanf(hex + 2*i, "%02x", &b);
        buf[i] = (uint8_t)b;
    }
    uint64_t nblocks = nbytes / 8;
    uint64_t r = xxh3_main_loop(acc, buf, nblocks);
    printf("%016llx\n", (unsigned long long)r);
    free(buf);
    return 0;
}
"#;

fn is_aarch64() -> bool {
    cfg!(target_arch = "aarch64")
}

fn has_cc() -> bool {
    Command::new("cc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn make_test_dir(test_name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("llvm2_xxh3_main_loop_{test_name}"));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("create test dir");
    dir
}

fn write_object_file(dir: &Path, filename: &str, bytes: &[u8]) -> PathBuf {
    let path = dir.join(filename);
    fs::write(&path, bytes).expect("write object file");
    path
}

fn write_c_driver(dir: &Path, filename: &str, source: &str) -> PathBuf {
    let path = dir.join(filename);
    fs::write(&path, source).expect("write C driver");
    path
}

fn link_with_cc(dir: &Path, driver_c: &Path, obj: &Path, output_name: &str) -> PathBuf {
    let binary = dir.join(output_name);
    let output = Command::new("cc")
        .arg("-o")
        .arg(&binary)
        .arg(driver_c)
        .arg(obj)
        .arg("-Wl,-no_pie")
        .output()
        .expect("run cc");
    if !output.status.success() {
        panic!(
            "link failed: {}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    binary
}

fn run_binary_with_args(binary: &Path, args: &[&str]) -> (i32, String) {
    let output = Command::new(binary).args(args).output().expect("run binary");
    (
        output.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&output.stdout).to_string(),
    )
}

fn cleanup(dir: &Path) {
    let _ = fs::remove_dir_all(dir);
}

fn compile_tmir(
    tmir_func: &tmir::Function,
    module: &tmir::Module,
    opt_level: OptLevel,
) -> Result<Vec<u8>, String> {
    let (lir_func, _proof) =
        llvm2_lower::translate_function(tmir_func, module).map_err(|e| format!("adapter: {e}"))?;
    let config = PipelineConfig {
        opt_level,
        emit_debug: false,
        ..Default::default()
    };
    let pipeline = Pipeline::new(config);
    pipeline
        .compile_function(&lir_func)
        .map_err(|e| format!("pipeline: {e}"))
}

fn assert_valid_macho(bytes: &[u8], ctx: &str) {
    assert!(bytes.len() >= 4, "{ctx}: object too small ({})", bytes.len());
    assert_eq!(&bytes[..4], &[0xCF, 0xFA, 0xED, 0xFE], "{ctx}: invalid Mach-O magic");
}

fn hex_encode(data: &[u8]) -> String {
    let mut hex = String::with_capacity(data.len() * 2);
    for byte in data {
        write!(&mut hex, "{byte:02x}").expect("write hex");
    }
    hex
}

fn sample_data(nbytes: usize) -> Vec<u8> {
    (0..nbytes)
        .map(|i| (i as u8).wrapping_mul(17).wrapping_add(3))
        .collect()
}

fn build_xxh3_main_loop() -> (tmir::Function, tmir::Module) {
    let mut mb = ModuleBuilder::new("xxh3_main_loop_fixture");
    let ty = mb.add_func_type(vec![Ty::I64, Ty::Ptr, Ty::I64], vec![Ty::I64]);
    let mut fb = mb.function("xxh3_main_loop", ty);

    let entry = fb.create_block();
    let acc = fb.add_block_param(entry, Ty::I64);
    let data = fb.add_block_param(entry, Ty::Ptr);
    let nblocks = fb.add_block_param(entry, Ty::I64);

    let loop_header = fb.create_block();
    let loop_acc = fb.add_block_param(loop_header, Ty::I64);
    let loop_i = fb.add_block_param(loop_header, Ty::I64);

    let loop_body = fb.create_block();

    let exit = fb.create_block();
    let exit_acc = fb.add_block_param(exit, Ty::I64);

    fb.switch_to_block(entry);
    let zero = fb.iconst(Ty::I64, 0);
    fb.br(loop_header, vec![acc, zero]);

    fb.switch_to_block(loop_header);
    let cond = fb.icmp(ICmpOp::Slt, Ty::I64, loop_i, nblocks);
    fb.condbr(cond, loop_body, vec![], exit, vec![loop_acc]);

    fb.switch_to_block(loop_body);
    let eight = fb.iconst(Ty::I64, 8);
    let byte_offset = fb.mul(Ty::I64, loop_i, eight);
    let block_ptr = fb.gep(Ty::I8, data, vec![byte_offset]);
    let block_val = fb.load(Ty::I64, block_ptr);
    let prime1 = fb.iconst(Ty::I64, XXH3_PRIME1 as i128);
    let mixed = fb.mul(Ty::I64, block_val, prime1);
    let tmp = fb.binop(BinOp::Xor, Ty::I64, loop_acc, mixed);
    let shl_amt = fb.iconst(Ty::I64, 31);
    let shr_amt = fb.iconst(Ty::I64, 33);
    let tmp_shl = fb.binop(BinOp::Shl, Ty::I64, tmp, shl_amt);
    let tmp_lshr = fb.binop(BinOp::LShr, Ty::I64, tmp, shr_amt);
    let rotated = fb.binop(BinOp::Or, Ty::I64, tmp_shl, tmp_lshr);
    let prime2 = fb.iconst(Ty::I64, XXH3_PRIME2 as i128);
    let next_acc = fb.mul(Ty::I64, rotated, prime2);
    let one = fb.iconst(Ty::I64, 1);
    let next_i = fb.add(Ty::I64, loop_i, one);
    fb.br(loop_header, vec![next_acc, next_i]);

    fb.switch_to_block(exit);
    fb.ret(vec![exit_acc]);

    fb.build();
    let module = mb.build();
    let func = module
        .functions
        .iter()
        .find(|f| f.name == "xxh3_main_loop")
        .expect("xxh3_main_loop function")
        .clone();
    (func, module)
}

fn xxh3_main_loop_ref(initial_acc: u64, data: &[u8]) -> u64 {
    let mut acc = initial_acc;
    let nblocks = data.len() / 8;
    for i in 0..nblocks {
        let off = i * 8;
        let block = u64::from_le_bytes(data[off..off + 8].try_into().unwrap());
        let tmp = acc ^ block.wrapping_mul(XXH3_PRIME1);
        acc = tmp.rotate_left(31).wrapping_mul(XXH3_PRIME2);
    }
    acc
}

fn run_xxh3_main_loop_case(test_name: &str, initial_acc: u64, data: &[u8]) {
    let dir = make_test_dir(test_name);
    let (func, module) = build_xxh3_main_loop();
    let obj_bytes = compile_tmir(&func, &module, OptLevel::O2).expect("O2 compile");
    assert_valid_macho(&obj_bytes, test_name);
    let obj_path = write_object_file(&dir, "xxh3_main_loop.o", &obj_bytes);
    let driver_path = write_c_driver(&dir, "driver.c", XXH3_MAIN_LOOP_DRIVER_C);
    let binary = link_with_cc(&dir, &driver_path, &obj_path, "xxh3_main_loop_test");
    let acc_hex = format!("{initial_acc:016x}");
    let data_hex = hex_encode(data);
    let (exit_code, stdout) = run_binary_with_args(&binary, &[&acc_hex, &data_hex]);
    assert_eq!(exit_code, 0, "{test_name}: binary failed with stdout {stdout:?}");
    let expected_stdout = format!("{:016x}\n", xxh3_main_loop_ref(initial_acc, data));
    assert_eq!(stdout, expected_stdout, "{test_name}: output mismatch");
    cleanup(&dir);
}

#[test]
fn test_xxh3_main_loop_ref_deterministic() {
    let initial_acc = 0x0123_4567_89AB_CDEF;
    let data = sample_data(64);
    assert_eq!(
        xxh3_main_loop_ref(initial_acc, &data),
        xxh3_main_loop_ref(initial_acc, &data)
    );
}

#[test]
fn test_xxh3_main_loop_ref_sensitive() {
    let initial_acc = 0x0123_4567_89AB_CDEF;
    let data_a = sample_data(32);
    let mut data_b = data_a.clone();
    data_b[15] ^= 0x5A;
    assert_ne!(
        xxh3_main_loop_ref(initial_acc, &data_a),
        xxh3_main_loop_ref(initial_acc, &data_b)
    );
}

#[test]
fn test_xxh3_main_loop_compiles_o0() {
    let (func, module) = build_xxh3_main_loop();
    let obj_bytes = compile_tmir(&func, &module, OptLevel::O0).expect("O0 compile");
    assert_valid_macho(&obj_bytes, "xxh3_main_loop O0");
    assert!(obj_bytes.len() > 100, "O0 object should be substantial");
}

#[test]
fn test_xxh3_main_loop_compiles_o2() {
    let (func, module) = build_xxh3_main_loop();
    let obj_bytes = compile_tmir(&func, &module, OptLevel::O2).expect("O2 compile");
    assert_valid_macho(&obj_bytes, "xxh3_main_loop O2");
    assert!(obj_bytes.len() > 100, "O2 object should be substantial");
}

#[test]
fn test_xxh3_main_loop_o2_differs_from_o0() {
    let (func, module) = build_xxh3_main_loop();
    let obj_o0 = compile_tmir(&func, &module, OptLevel::O0).expect("O0 compile");
    let obj_o2 = compile_tmir(&func, &module, OptLevel::O2).expect("O2 compile");
    assert_ne!(obj_o0, obj_o2, "O2 output should differ from O0");
}

#[test]
fn test_xxh3_main_loop_determinism_o2() {
    let (func, module) = build_xxh3_main_loop();
    let obj_a = compile_tmir(&func, &module, OptLevel::O2).expect("O2 compile A");
    let obj_b = compile_tmir(&func, &module, OptLevel::O2).expect("O2 compile B");
    assert_eq!(obj_a, obj_b, "O2 output must be byte-identical");
}

#[test]
fn test_xxh3_main_loop_e2e_single_block() {
    if !is_aarch64() || !has_cc() {
        return;
    }
    run_xxh3_main_loop_case(
        "single_block",
        0x243F_6A88_85A3_08D3,
        &sample_data(8),
    );
}

#[test]
fn test_xxh3_main_loop_e2e_four_blocks() {
    if !is_aarch64() || !has_cc() {
        return;
    }
    run_xxh3_main_loop_case(
        "four_blocks",
        0x1319_8A2E_0370_7344,
        &sample_data(32),
    );
}

#[test]
fn test_xxh3_main_loop_e2e_sixteen_blocks() {
    if !is_aarch64() || !has_cc() {
        return;
    }
    run_xxh3_main_loop_case(
        "sixteen_blocks",
        0xA409_3822_299F_31D0,
        &sample_data(128),
    );
}

#[test]
fn test_xxh3_main_loop_e2e_zero_blocks() {
    if !is_aarch64() || !has_cc() {
        return;
    }
    run_xxh3_main_loop_case("zero_blocks", 0x082E_FA98_EC4E_6C89, &[]);
}
