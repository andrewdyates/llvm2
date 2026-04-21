// llvm2-codegen/benches/compile_speed.rs - Compilation speed benchmark suite
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Comprehensive compilation speed benchmarks measuring all pipeline phases:
//   - ISel throughput (tMIR -> LIR via translate_module)
//   - Optimization pipeline throughput per function
//   - Register allocation time
//   - Full pipeline end-to-end (tMIR -> .o via Compiler API)
//   - Parallel vs serial compilation comparison
//
// Run with: cargo bench -p llvm2-codegen --bench compile_speed
//
// Part of #315 — Compilation speed benchmarks

use std::time::{Duration, Instant};

use tmir::{BinOp, ICmpOp, Ty};
use tmir_build::ModuleBuilder;

use llvm2_codegen::compiler::{Compiler, CompilerConfig, CompilerTraceLevel};
use llvm2_codegen::pipeline::{OptLevel, ir_to_regalloc, apply_regalloc};
use llvm2_opt::pipeline::{OptLevel as OptOptLevel, OptimizationPipeline};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Number of warmup iterations before measurement.
const WARMUP_ITERS: u32 = 3;

/// Minimum measurement iterations.
const BENCH_ITERS: u32 = 50;

/// Minimum benchmark duration to ensure stable timings.
const MIN_BENCH_DURATION: Duration = Duration::from_millis(300);

// ---------------------------------------------------------------------------
// Test module builders using tmir_build::ModuleBuilder
// ---------------------------------------------------------------------------

/// Build a small module: 1 function with ~5 operations.
/// f(a: i64, b: i64) -> i64 = (a + b) - (a * b) + 42
fn build_small_module() -> tmir::Module {
    let mut mb = ModuleBuilder::new("bench_small");
    let ty = mb.add_func_type(vec![Ty::I64, Ty::I64], vec![Ty::I64]);
    let mut fb = mb.function("small_arith", ty);
    let entry = fb.create_block();
    let a = fb.add_block_param(entry, Ty::I64);
    let b = fb.add_block_param(entry, Ty::I64);
    fb.switch_to_block(entry);

    let sum = fb.add(Ty::I64, a, b);
    let prod = fb.mul(Ty::I64, a, b);
    let diff = fb.sub(Ty::I64, sum, prod);
    let c42 = fb.iconst(Ty::I64, 42);
    let result = fb.add(Ty::I64, diff, c42);
    fb.ret(vec![result]);
    fb.build();
    mb.build()
}

/// Build a medium module: 3 functions each with ~10 operations.
fn build_medium_module() -> tmir::Module {
    let mut mb = ModuleBuilder::new("bench_medium");

    // Function 1: polynomial evaluation f(x) = x^2 + 3x + 7
    {
        let ty = mb.add_func_type(vec![Ty::I64], vec![Ty::I64]);
        let mut fb = mb.function("poly_eval", ty);
        let entry = fb.create_block();
        let x = fb.add_block_param(entry, Ty::I64);
        fb.switch_to_block(entry);

        let x2 = fb.mul(Ty::I64, x, x);
        let c3 = fb.iconst(Ty::I64, 3);
        let x3 = fb.mul(Ty::I64, x, c3);
        let sum1 = fb.add(Ty::I64, x2, x3);
        let c7 = fb.iconst(Ty::I64, 7);
        let result = fb.add(Ty::I64, sum1, c7);
        let c1 = fb.iconst(Ty::I64, 1);
        let adj = fb.sub(Ty::I64, result, c1);
        let c2 = fb.iconst(Ty::I64, 2);
        let final_val = fb.add(Ty::I64, adj, c2);
        fb.ret(vec![final_val]);
        fb.build();
    }

    // Function 2: max(a, b) using select
    {
        let ty = mb.add_func_type(vec![Ty::I64, Ty::I64], vec![Ty::I64]);
        let mut fb = mb.function("max_val", ty);
        let entry = fb.create_block();
        let a = fb.add_block_param(entry, Ty::I64);
        let b = fb.add_block_param(entry, Ty::I64);
        fb.switch_to_block(entry);

        let cond = fb.icmp(ICmpOp::Sgt, Ty::I64, a, b);
        let result = fb.select(Ty::I64, cond, a, b);
        let c1 = fb.iconst(Ty::I64, 1);
        let inc = fb.add(Ty::I64, result, c1);
        let c10 = fb.iconst(Ty::I64, 10);
        let bounded = fb.sub(Ty::I64, inc, c10);
        let cond2 = fb.icmp(ICmpOp::Sgt, Ty::I64, bounded, c1);
        let final_val = fb.select(Ty::I64, cond2, bounded, c1);
        fb.ret(vec![final_val]);
        fb.build();
    }

    // Function 3: abs(a - b) using select
    {
        let ty = mb.add_func_type(vec![Ty::I64, Ty::I64], vec![Ty::I64]);
        let mut fb = mb.function("abs_diff", ty);
        let entry = fb.create_block();
        let a = fb.add_block_param(entry, Ty::I64);
        let b = fb.add_block_param(entry, Ty::I64);
        fb.switch_to_block(entry);

        let diff = fb.sub(Ty::I64, a, b);
        let c0 = fb.iconst(Ty::I64, 0);
        let neg = fb.sub(Ty::I64, c0, diff);
        let cond = fb.icmp(ICmpOp::Sge, Ty::I64, diff, c0);
        let abs_val = fb.select(Ty::I64, cond, diff, neg);
        let c1 = fb.iconst(Ty::I64, 1);
        let result = fb.add(Ty::I64, abs_val, c1);
        let c2 = fb.iconst(Ty::I64, 2);
        let final_val = fb.mul(Ty::I64, result, c2);
        fb.ret(vec![final_val]);
        fb.build();
    }

    mb.build()
}

/// Build a large module: 5 functions each with ~20 operations.
fn build_large_module() -> tmir::Module {
    let mut mb = ModuleBuilder::new("bench_large");

    // Function 1: extended polynomial f(x, y) = x^2 + y^2 + 2xy - x - y + 1
    {
        let ty = mb.add_func_type(vec![Ty::I64, Ty::I64], vec![Ty::I64]);
        let mut fb = mb.function("extended_poly", ty);
        let entry = fb.create_block();
        let x = fb.add_block_param(entry, Ty::I64);
        let y = fb.add_block_param(entry, Ty::I64);
        fb.switch_to_block(entry);

        let x2 = fb.mul(Ty::I64, x, x);
        let y2 = fb.mul(Ty::I64, y, y);
        let xy = fb.mul(Ty::I64, x, y);
        let c2 = fb.iconst(Ty::I64, 2);
        let xy2 = fb.mul(Ty::I64, xy, c2);
        let sum1 = fb.add(Ty::I64, x2, y2);
        let sum2 = fb.add(Ty::I64, sum1, xy2);
        let sub1 = fb.sub(Ty::I64, sum2, x);
        let sub2 = fb.sub(Ty::I64, sub1, y);
        let c1 = fb.iconst(Ty::I64, 1);
        let result = fb.add(Ty::I64, sub2, c1);
        let c3 = fb.iconst(Ty::I64, 3);
        let adj = fb.add(Ty::I64, result, c3);
        let c5 = fb.iconst(Ty::I64, 5);
        let adj2 = fb.sub(Ty::I64, adj, c5);
        let c7 = fb.iconst(Ty::I64, 7);
        let adj3 = fb.add(Ty::I64, adj2, c7);
        let adj4 = fb.sub(Ty::I64, adj3, c1);
        let adj5 = fb.add(Ty::I64, adj4, c2);
        let final_val = fb.sub(Ty::I64, adj5, c3);
        fb.ret(vec![final_val]);
        fb.build();
    }

    // Function 2: clamp(val, lo, hi)
    {
        let ty = mb.add_func_type(vec![Ty::I64, Ty::I64, Ty::I64], vec![Ty::I64]);
        let mut fb = mb.function("clamp", ty);
        let entry = fb.create_block();
        let val = fb.add_block_param(entry, Ty::I64);
        let lo = fb.add_block_param(entry, Ty::I64);
        let hi = fb.add_block_param(entry, Ty::I64);
        fb.switch_to_block(entry);

        let cond_lo = fb.icmp(ICmpOp::Slt, Ty::I64, val, lo);
        let clamped_lo = fb.select(Ty::I64, cond_lo, lo, val);
        let cond_hi = fb.icmp(ICmpOp::Sgt, Ty::I64, clamped_lo, hi);
        let clamped = fb.select(Ty::I64, cond_hi, hi, clamped_lo);
        let c1 = fb.iconst(Ty::I64, 1);
        let inc = fb.add(Ty::I64, clamped, c1);
        let dec = fb.sub(Ty::I64, inc, c1);
        let c0 = fb.iconst(Ty::I64, 0);
        let cond_pos = fb.icmp(ICmpOp::Sge, Ty::I64, dec, c0);
        let abs_val = fb.select(Ty::I64, cond_pos, dec, c0);
        let c2 = fb.iconst(Ty::I64, 2);
        let doubled = fb.mul(Ty::I64, abs_val, c2);
        let c10 = fb.iconst(Ty::I64, 10);
        let bounded = fb.sub(Ty::I64, doubled, c10);
        let cond_bound = fb.icmp(ICmpOp::Sgt, Ty::I64, bounded, c0);
        let result = fb.select(Ty::I64, cond_bound, bounded, c0);
        let c3 = fb.iconst(Ty::I64, 3);
        let final_val = fb.add(Ty::I64, result, c3);
        fb.ret(vec![final_val]);
        fb.build();
    }

    // Function 3: sum of squares sum_sq(a, b) = a^2 + b^2 + (a+b)^2
    {
        let ty = mb.add_func_type(vec![Ty::I64, Ty::I64], vec![Ty::I64]);
        let mut fb = mb.function("sum_squares", ty);
        let entry = fb.create_block();
        let a = fb.add_block_param(entry, Ty::I64);
        let b = fb.add_block_param(entry, Ty::I64);
        fb.switch_to_block(entry);

        let a2 = fb.mul(Ty::I64, a, a);
        let b2 = fb.mul(Ty::I64, b, b);
        let ab = fb.add(Ty::I64, a, b);
        let ab2 = fb.mul(Ty::I64, ab, ab);
        let s1 = fb.add(Ty::I64, a2, b2);
        let s2 = fb.add(Ty::I64, s1, ab2);
        let c4 = fb.iconst(Ty::I64, 4);
        let d = fb.sub(Ty::I64, s2, c4);
        let c2 = fb.iconst(Ty::I64, 2);
        let m = fb.mul(Ty::I64, d, c2);
        let c1 = fb.iconst(Ty::I64, 1);
        let adj = fb.add(Ty::I64, m, c1);
        let c8 = fb.iconst(Ty::I64, 8);
        let s = fb.sub(Ty::I64, adj, c8);
        let c3 = fb.iconst(Ty::I64, 3);
        let a2b = fb.add(Ty::I64, s, c3);
        let c6 = fb.iconst(Ty::I64, 6);
        let final_val = fb.sub(Ty::I64, a2b, c6);
        fb.ret(vec![final_val]);
        fb.build();
    }

    // Function 4: dot product approximation dot(a1, b1, a2, b2) = a1*b1 + a2*b2
    {
        let ty = mb.add_func_type(vec![Ty::I64, Ty::I64, Ty::I64, Ty::I64], vec![Ty::I64]);
        let mut fb = mb.function("dot_approx", ty);
        let entry = fb.create_block();
        let a1 = fb.add_block_param(entry, Ty::I64);
        let b1 = fb.add_block_param(entry, Ty::I64);
        let a2 = fb.add_block_param(entry, Ty::I64);
        let b2 = fb.add_block_param(entry, Ty::I64);
        fb.switch_to_block(entry);

        let prod1 = fb.mul(Ty::I64, a1, b1);
        let prod2 = fb.mul(Ty::I64, a2, b2);
        let dot = fb.add(Ty::I64, prod1, prod2);
        let c1 = fb.iconst(Ty::I64, 1);
        let inc = fb.add(Ty::I64, dot, c1);
        let c2 = fb.iconst(Ty::I64, 2);
        let scaled = fb.mul(Ty::I64, inc, c2);
        let half = fb.sub(Ty::I64, scaled, dot);
        let c3 = fb.iconst(Ty::I64, 3);
        let adj = fb.add(Ty::I64, half, c3);
        let c4 = fb.iconst(Ty::I64, 4);
        let s = fb.sub(Ty::I64, adj, c4);
        let c5 = fb.iconst(Ty::I64, 5);
        let a3 = fb.add(Ty::I64, s, c5);
        let neg = fb.sub(Ty::I64, a3, c1);
        let c7 = fb.iconst(Ty::I64, 7);
        let m = fb.mul(Ty::I64, neg, c7);
        let c9 = fb.iconst(Ty::I64, 9);
        let final_val = fb.sub(Ty::I64, m, c9);
        fb.ret(vec![final_val]);
        fb.build();
    }

    // Function 5: min/max chain
    {
        let ty = mb.add_func_type(vec![Ty::I64, Ty::I64, Ty::I64], vec![Ty::I64]);
        let mut fb = mb.function("min_max_chain", ty);
        let entry = fb.create_block();
        let a = fb.add_block_param(entry, Ty::I64);
        let b = fb.add_block_param(entry, Ty::I64);
        let c = fb.add_block_param(entry, Ty::I64);
        fb.switch_to_block(entry);

        let cond1 = fb.icmp(ICmpOp::Sgt, Ty::I64, a, b);
        let max_ab = fb.select(Ty::I64, cond1, a, b);
        let cond2 = fb.icmp(ICmpOp::Sgt, Ty::I64, max_ab, c);
        let max_abc = fb.select(Ty::I64, cond2, max_ab, c);
        let cond3 = fb.icmp(ICmpOp::Slt, Ty::I64, a, b);
        let min_ab = fb.select(Ty::I64, cond3, a, b);
        let cond4 = fb.icmp(ICmpOp::Slt, Ty::I64, min_ab, c);
        let min_abc = fb.select(Ty::I64, cond4, min_ab, c);
        let range = fb.sub(Ty::I64, max_abc, min_abc);
        let c2 = fb.iconst(Ty::I64, 2);
        let half_range = fb.sub(Ty::I64, range, c2);
        let sum = fb.add(Ty::I64, max_abc, min_abc);
        let mid = fb.sub(Ty::I64, sum, half_range);
        let c1 = fb.iconst(Ty::I64, 1);
        let result = fb.add(Ty::I64, mid, c1);
        let c3 = fb.iconst(Ty::I64, 3);
        let final_val = fb.mul(Ty::I64, result, c3);
        fb.ret(vec![final_val]);
        fb.build();
    }

    mb.build()
}

/// Build an xlarge module: 10 functions, each with ~15 operations.
/// Demonstrates parallel scaling with more functions to amortize overhead.
fn build_xlarge_module() -> tmir::Module {
    let mut mb = ModuleBuilder::new("bench_xlarge");

    for i in 0..10u64 {
        let ty = mb.add_func_type(vec![Ty::I64, Ty::I64], vec![Ty::I64]);
        let name = format!("xlarge_fn_{}", i);
        let mut fb = mb.function(&name, ty);
        let entry = fb.create_block();
        let a = fb.add_block_param(entry, Ty::I64);
        let b = fb.add_block_param(entry, Ty::I64);
        fb.switch_to_block(entry);

        // Parametric polynomial: result = a^2 + b^2 + (i+1)*a*b - i*a + (i+2)*b + i^2
        let a2 = fb.mul(Ty::I64, a, a);
        let b2 = fb.mul(Ty::I64, b, b);
        let ab = fb.mul(Ty::I64, a, b);
        let ci1 = fb.iconst(Ty::I64, (i + 1) as i128);
        let ab_scaled = fb.mul(Ty::I64, ab, ci1);
        let sum1 = fb.add(Ty::I64, a2, b2);
        let sum2 = fb.add(Ty::I64, sum1, ab_scaled);
        let ci = fb.iconst(Ty::I64, i as i128);
        let ia = fb.mul(Ty::I64, ci, a);
        let sub1 = fb.sub(Ty::I64, sum2, ia);
        let ci2 = fb.iconst(Ty::I64, (i + 2) as i128);
        let ib = fb.mul(Ty::I64, ci2, b);
        let sum3 = fb.add(Ty::I64, sub1, ib);
        let ci_sq = fb.iconst(Ty::I64, (i * i) as i128);
        let result = fb.add(Ty::I64, sum3, ci_sq);
        fb.ret(vec![result]);
        fb.build();
    }

    mb.build()
}

/// Build a stress-test module: 20 functions, each with ~12 operations.
/// Tests parallel scaling at higher function counts where rayon overhead
/// is well-amortized across the thread pool.
fn build_stress_module() -> tmir::Module {
    let mut mb = ModuleBuilder::new("bench_stress");

    for i in 0..20u64 {
        let ty = mb.add_func_type(vec![Ty::I64, Ty::I64], vec![Ty::I64]);
        let name = format!("stress_fn_{}", i);
        let mut fb = mb.function(&name, ty);
        let entry = fb.create_block();
        let a = fb.add_block_param(entry, Ty::I64);
        let b = fb.add_block_param(entry, Ty::I64);
        fb.switch_to_block(entry);

        // Parametric chain: ((a + c_i) * b - (a * c_j)) + c_k
        let ci = fb.iconst(Ty::I64, (i * 3 + 1) as i128);
        let t1 = fb.add(Ty::I64, a, ci);
        let t2 = fb.mul(Ty::I64, t1, b);
        let cj = fb.iconst(Ty::I64, (i * 2 + 5) as i128);
        let t3 = fb.mul(Ty::I64, a, cj);
        let t4 = fb.sub(Ty::I64, t2, t3);
        let ck = fb.iconst(Ty::I64, (i * i + 7) as i128);
        let t5 = fb.add(Ty::I64, t4, ck);
        let t6 = fb.mul(Ty::I64, t5, b);
        let cl = fb.iconst(Ty::I64, (i + 10) as i128);
        let t7 = fb.sub(Ty::I64, t6, cl);
        let t8 = fb.add(Ty::I64, t7, a);
        fb.ret(vec![t8]);
        fb.build();
    }

    mb.build()
}

/// Build the xxh3 main-loop primitive module (Part of #343).
///
/// Shape: one function `xxh3_main_loop(acc, data, nblocks) -> acc` implementing
/// a counted loop over 8-byte chunks, with a rotate-mul-xor round per block.
/// This is the hot path for tla2's compiled state fingerprinting — a realistic
/// loop-shaped kernel with a live-through block-param accumulator, 64-bit
/// load/multiply/xor/shift/or, and a counted increment. It stresses regalloc
/// and instruction scheduling differently from the straight-line arithmetic
/// kernels above. See crates/llvm2-codegen/tests/xxh3_main_loop_tmir.rs.
fn build_xxh3_main_loop_module() -> tmir::Module {
    const XXH3_PRIME1: u64 = 0x9E37_79B1_85EB_CA87;
    const XXH3_PRIME2: u64 = 0xC2B2_AE3D_27D4_EB4F;

    let mut mb = ModuleBuilder::new("bench_xxh3_main_loop");
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
    mb.build()
}

// ---------------------------------------------------------------------------
// Timing utilities
// ---------------------------------------------------------------------------

/// Determine iteration count based on a probe run.
fn determine_iters(probe_fn: &dyn Fn()) -> u32 {
    let probe_start = Instant::now();
    for _ in 0..5 {
        probe_fn();
    }
    let probe_elapsed = probe_start.elapsed();
    let per_iter_ns = probe_elapsed.as_nanos() / 5;
    let min_iters = if per_iter_ns > 0 {
        (MIN_BENCH_DURATION.as_nanos() / per_iter_ns) as u32
    } else {
        BENCH_ITERS
    };
    min_iters.max(BENCH_ITERS)
}

fn format_duration(d: Duration, iters: u32) -> String {
    let per_iter_ns = d.as_nanos() / iters as u128;
    if per_iter_ns < 1_000 {
        format!("{:>7} ns", per_iter_ns)
    } else if per_iter_ns < 1_000_000 {
        format!("{:>7.1} us", per_iter_ns as f64 / 1_000.0)
    } else {
        format!("{:>7.2} ms", per_iter_ns as f64 / 1_000_000.0)
    }
}

fn format_throughput(count: usize, duration: Duration, iters: u32) -> String {
    let total = count as f64 * iters as f64;
    let per_sec = total / duration.as_secs_f64();
    if per_sec > 1_000_000.0 {
        format!("{:.2}M/s", per_sec / 1_000_000.0)
    } else if per_sec > 1_000.0 {
        format!("{:.1}K/s", per_sec / 1_000.0)
    } else {
        format!("{:.0}/s", per_sec)
    }
}

// ---------------------------------------------------------------------------
// Benchmark: ISel throughput (tMIR -> LIR via translate_module)
// ---------------------------------------------------------------------------

struct ISelResult {
    module_name: &'static str,
    func_count: usize,
    duration: Duration,
    iters: u32,
}

fn bench_isel(module: &tmir::Module, name: &'static str) -> ISelResult {
    let func_count = module.functions.len();

    // Warmup
    for _ in 0..WARMUP_ITERS {
        let _ = llvm2_lower::translate_module(module);
    }

    let iters = determine_iters(&|| {
        let _ = llvm2_lower::translate_module(module);
    });

    let start = Instant::now();
    for _ in 0..iters {
        let _ = llvm2_lower::translate_module(module);
    }
    let duration = start.elapsed();

    ISelResult { module_name: name, func_count, duration, iters }
}

// ---------------------------------------------------------------------------
// Benchmark: Optimization pipeline throughput
// ---------------------------------------------------------------------------

struct OptResult {
    module_name: &'static str,
    func_count: usize,
    duration: Duration,
    iters: u32,
    opt_level: &'static str,
}

fn bench_optimization(module: &tmir::Module, name: &'static str, opt_level: OptOptLevel) -> OptResult {
    // First translate to get LIR functions
    let lir_funcs = llvm2_lower::translate_module(module).expect("translate_module failed");

    // Run ISel to get post-ISel IR functions (before optimization).
    let ir_funcs_fresh: Vec<_> = lir_funcs.iter()
        .filter_map(|(lir_func, _)| {
            use llvm2_lower::isel::InstructionSelector;
            let sig = llvm2_lower::function::Signature {
                params: lir_func.signature.params.clone(),
                returns: lir_func.signature.returns.clone(),
            };
            let mut isel = InstructionSelector::new(lir_func.name.clone(), sig.clone());
            isel.set_stack_slots(lir_func.stack_slots.clone());
            isel.seed_value_types(&lir_func.value_types);
            isel.seed_pure_callees(&lir_func.pure_callees);
            isel.lower_formal_arguments(&sig, lir_func.entry_block).ok()?;

            let mut block_order: Vec<_> = lir_func.blocks.keys().copied().collect();
            block_order.sort_by_key(|b| b.0);

            for &block in &block_order {
                let bb = &lir_func.blocks[&block];
                if block != lir_func.entry_block && !bb.params.is_empty() {
                    isel.define_block_params(&bb.params);
                }
                isel.select_block(block, &bb.instructions).ok()?;
            }

            let isel_func = isel.finalize();
            Some(isel_func.to_ir_func())
        })
        .collect();

    let func_count = ir_funcs_fresh.len();
    if func_count == 0 {
        return OptResult {
            module_name: name,
            func_count: 0,
            duration: Duration::ZERO,
            iters: 0,
            opt_level: opt_level_name(opt_level),
        };
    }

    // Warmup
    for _ in 0..WARMUP_ITERS {
        for f in &ir_funcs_fresh {
            let mut clone = f.clone();
            let opt_pipeline = OptimizationPipeline::new(opt_level);
            let _ = opt_pipeline.run(&mut clone);
        }
    }

    let iters = determine_iters(&|| {
        for f in &ir_funcs_fresh {
            let mut clone = f.clone();
            let opt_pipeline = OptimizationPipeline::new(opt_level);
            let _ = opt_pipeline.run(&mut clone);
        }
    });

    let start = Instant::now();
    for _ in 0..iters {
        for f in &ir_funcs_fresh {
            let mut clone = f.clone();
            let opt_pipeline = OptimizationPipeline::new(opt_level);
            let _ = opt_pipeline.run(&mut clone);
        }
    }
    let duration = start.elapsed();

    OptResult {
        module_name: name,
        func_count,
        duration,
        iters,
        opt_level: opt_level_name(opt_level),
    }
}

fn opt_level_name(level: OptOptLevel) -> &'static str {
    match level {
        OptOptLevel::O0 => "O0",
        OptOptLevel::O1 => "O1",
        OptOptLevel::O2 => "O2",
        OptOptLevel::O3 => "O3",
        _ => "Other",
    }
}

// ---------------------------------------------------------------------------
// Benchmark: Register allocation time
// ---------------------------------------------------------------------------

struct RegAllocResult {
    module_name: &'static str,
    func_count: usize,
    duration: Duration,
    iters: u32,
}

fn bench_regalloc(module: &tmir::Module, name: &'static str) -> RegAllocResult {
    // Translate and run ISel + opt to get functions ready for regalloc
    let lir_funcs = llvm2_lower::translate_module(module).expect("translate_module failed");

    let ir_funcs_for_ra: Vec<_> = lir_funcs.iter()
        .filter_map(|(lir_func, _)| {
            use llvm2_lower::isel::InstructionSelector;
            let sig = llvm2_lower::function::Signature {
                params: lir_func.signature.params.clone(),
                returns: lir_func.signature.returns.clone(),
            };
            let mut isel = InstructionSelector::new(lir_func.name.clone(), sig.clone());
            isel.set_stack_slots(lir_func.stack_slots.clone());
            isel.seed_value_types(&lir_func.value_types);
            isel.seed_pure_callees(&lir_func.pure_callees);
            isel.lower_formal_arguments(&sig, lir_func.entry_block).ok()?;

            let mut block_order: Vec<_> = lir_func.blocks.keys().copied().collect();
            block_order.sort_by_key(|b| b.0);

            for &block in &block_order {
                let bb = &lir_func.blocks[&block];
                if block != lir_func.entry_block && !bb.params.is_empty() {
                    isel.define_block_params(&bb.params);
                }
                isel.select_block(block, &bb.instructions).ok()?;
            }

            let isel_func = isel.finalize();
            let mut ir_func = isel_func.to_ir_func();
            // Run optimization first (regalloc operates on optimized IR)
            let opt_pipeline = OptimizationPipeline::new(OptOptLevel::O2);
            let _ = opt_pipeline.run(&mut ir_func);
            Some(ir_func)
        })
        .collect();

    let func_count = ir_funcs_for_ra.len();
    if func_count == 0 {
        return RegAllocResult {
            module_name: name,
            func_count: 0,
            duration: Duration::ZERO,
            iters: 0,
        };
    }

    // Warmup
    for _ in 0..WARMUP_ITERS {
        for f in &ir_funcs_for_ra {
            let mut clone = f.clone();
            let ra_func = ir_to_regalloc(&clone);
            if let Ok(mut ra) = ra_func {
                let config = llvm2_regalloc::AllocConfig::default_aarch64();
                if let Ok(result) = llvm2_regalloc::allocate(&mut ra, &config) {
                    apply_regalloc(&mut clone, &result.allocation);
                }
            }
        }
    }

    let iters = determine_iters(&|| {
        for f in &ir_funcs_for_ra {
            let mut clone = f.clone();
            let ra_func = ir_to_regalloc(&clone);
            if let Ok(mut ra) = ra_func {
                let config = llvm2_regalloc::AllocConfig::default_aarch64();
                if let Ok(result) = llvm2_regalloc::allocate(&mut ra, &config) {
                    apply_regalloc(&mut clone, &result.allocation);
                }
            }
        }
    });

    let start = Instant::now();
    for _ in 0..iters {
        for f in &ir_funcs_for_ra {
            let mut clone = f.clone();
            let ra_func = ir_to_regalloc(&clone);
            if let Ok(mut ra) = ra_func {
                let config = llvm2_regalloc::AllocConfig::default_aarch64();
                if let Ok(result) = llvm2_regalloc::allocate(&mut ra, &config) {
                    apply_regalloc(&mut clone, &result.allocation);
                }
            }
        }
    }
    let duration = start.elapsed();

    RegAllocResult { module_name: name, func_count, duration, iters }
}

// ---------------------------------------------------------------------------
// Benchmark: Full pipeline end-to-end (tMIR -> .o)
// ---------------------------------------------------------------------------

struct E2EResult {
    module_name: &'static str,
    func_count: usize,
    duration: Duration,
    iters: u32,
    code_size: usize,
    parallel: bool,
}

fn bench_e2e(module: &tmir::Module, name: &'static str, parallel: bool) -> E2EResult {
    let func_count = module.functions.len();

    let config = CompilerConfig {
        opt_level: OptLevel::O2,
        target: llvm2_codegen::Target::Aarch64,
        emit_proofs: false,
        trace_level: CompilerTraceLevel::None,
        emit_debug: false,
        parallel,
        cegis_superopt_budget_sec: None,
    };
    let compiler = Compiler::new(config);

    // Warmup
    let mut code_size = 0;
    for _ in 0..WARMUP_ITERS {
        if let Ok(result) = compiler.compile(module) {
            code_size = result.object_code.len();
        }
    }

    let iters = determine_iters(&|| {
        let _ = compiler.compile(module);
    });

    let start = Instant::now();
    for _ in 0..iters {
        let _ = compiler.compile(module);
    }
    let duration = start.elapsed();

    E2EResult { module_name: name, func_count, duration, iters, code_size, parallel }
}

// ---------------------------------------------------------------------------
// Report formatting
// ---------------------------------------------------------------------------

fn print_report(
    isel_results: &[ISelResult],
    opt_results: &[OptResult],
    regalloc_results: &[RegAllocResult],
    e2e_results: &[E2EResult],
) {
    println!();
    println!("==============================================================================");
    println!("  LLVM2 Compilation Speed Benchmark Report");
    println!("==============================================================================");
    println!();
    println!("  Architecture: AArch64 (Apple Silicon)");
    println!("  Optimization: O2");
    println!("  Pipeline: tMIR -> ISel -> Opt -> RegAlloc -> Frame -> Encode -> Mach-O");
    println!();

    // ISel throughput
    println!("  --- ISel Throughput (tMIR -> LIR) ---");
    println!();
    println!("  {:<16} {:>5} {:>12} {:>12} {:>8}",
        "Module", "Funcs", "Time/module", "Funcs/sec", "Iters");
    println!("  {:-<16} {:->5} {:->12} {:->12} {:->8}",
        "", "", "", "", "");
    for r in isel_results {
        println!("  {:<16} {:>5} {:>12} {:>12} {:>8}",
            r.module_name,
            r.func_count,
            format_duration(r.duration, r.iters),
            format_throughput(r.func_count, r.duration, r.iters),
            r.iters);
    }
    println!();

    // Optimization throughput
    println!("  --- Optimization Pipeline Throughput ---");
    println!();
    println!("  {:<16} {:>5} {:>5} {:>12} {:>12} {:>8}",
        "Module", "Funcs", "Level", "Time/batch", "Funcs/sec", "Iters");
    println!("  {:-<16} {:->5} {:->5} {:->12} {:->12} {:->8}",
        "", "", "", "", "", "");
    for r in opt_results {
        println!("  {:<16} {:>5} {:>5} {:>12} {:>12} {:>8}",
            r.module_name,
            r.func_count,
            r.opt_level,
            format_duration(r.duration, r.iters),
            format_throughput(r.func_count, r.duration, r.iters),
            r.iters);
    }
    println!();

    // Register allocation
    println!("  --- Register Allocation Time ---");
    println!();
    println!("  {:<16} {:>5} {:>12} {:>12} {:>8}",
        "Module", "Funcs", "Time/batch", "Funcs/sec", "Iters");
    println!("  {:-<16} {:->5} {:->12} {:->12} {:->8}",
        "", "", "", "", "");
    for r in regalloc_results {
        println!("  {:<16} {:>5} {:>12} {:>12} {:>8}",
            r.module_name,
            r.func_count,
            format_duration(r.duration, r.iters),
            format_throughput(r.func_count, r.duration, r.iters),
            r.iters);
    }
    println!();

    // End-to-end
    println!("  --- Full Pipeline End-to-End (tMIR -> .o) ---");
    println!();
    println!("  {:<16} {:>5} {:>8} {:>12} {:>12} {:>8} {:>8}",
        "Module", "Funcs", "Parallel", "Time/module", "Funcs/sec", "ObjSize", "Iters");
    println!("  {:-<16} {:->5} {:->8} {:->12} {:->12} {:->8} {:->8}",
        "", "", "", "", "", "", "");
    for r in e2e_results {
        println!("  {:<16} {:>5} {:>8} {:>12} {:>12} {:>7}B {:>8}",
            r.module_name,
            r.func_count,
            if r.parallel { "yes" } else { "no" },
            format_duration(r.duration, r.iters),
            format_throughput(r.func_count, r.duration, r.iters),
            r.code_size,
            r.iters);
    }
    println!();

    // Parallel vs serial speedup
    println!("  --- Parallel vs Serial Speedup ---");
    println!();
    let serial: Vec<_> = e2e_results.iter().filter(|r| !r.parallel).collect();
    let parallel: Vec<_> = e2e_results.iter().filter(|r| r.parallel).collect();
    for (s, p) in serial.iter().zip(parallel.iter()) {
        if s.module_name == p.module_name {
            let serial_ns = s.duration.as_nanos() as f64 / s.iters as f64;
            let parallel_ns = p.duration.as_nanos() as f64 / p.iters as f64;
            let speedup = if parallel_ns > 0.0 { serial_ns / parallel_ns } else { 0.0 };
            println!("  {:<16}  serial: {:>12}  parallel: {:>12}  speedup: {:.2}x",
                s.module_name,
                format_duration(s.duration, s.iters),
                format_duration(p.duration, p.iters),
                speedup);
        }
    }

    println!();
    println!("==============================================================================");
    println!();
}

// ---------------------------------------------------------------------------
// Main benchmark entry point
// ---------------------------------------------------------------------------

fn main() {
    eprintln!("  Building test modules...");
    let small = build_small_module();
    let medium = build_medium_module();
    let large = build_large_module();
    let xlarge = build_xlarge_module();
    let stress = build_stress_module();
    let xxh3 = build_xxh3_main_loop_module();

    // --- ISel benchmarks ---
    eprintln!("  Benchmarking ISel...");
    let isel_results = vec![
        bench_isel(&small, "small (1fn)"),
        bench_isel(&medium, "medium (3fn)"),
        bench_isel(&large, "large (5fn)"),
        bench_isel(&xlarge, "xlarge (10fn)"),
        bench_isel(&stress, "stress (20fn)"),
        bench_isel(&xxh3, "xxh3 (loop)"),
    ];

    // --- Optimization benchmarks ---
    eprintln!("  Benchmarking Optimization...");
    let opt_results = vec![
        bench_optimization(&small, "small (1fn)", OptOptLevel::O0),
        bench_optimization(&small, "small (1fn)", OptOptLevel::O2),
        bench_optimization(&medium, "medium (3fn)", OptOptLevel::O2),
        bench_optimization(&large, "large (5fn)", OptOptLevel::O2),
        bench_optimization(&xlarge, "xlarge (10fn)", OptOptLevel::O2),
        bench_optimization(&stress, "stress (20fn)", OptOptLevel::O2),
        bench_optimization(&xxh3, "xxh3 (loop)", OptOptLevel::O0),
        bench_optimization(&xxh3, "xxh3 (loop)", OptOptLevel::O2),
    ];

    // --- RegAlloc benchmarks ---
    eprintln!("  Benchmarking RegAlloc...");
    let regalloc_results = vec![
        bench_regalloc(&small, "small (1fn)"),
        bench_regalloc(&medium, "medium (3fn)"),
        bench_regalloc(&large, "large (5fn)"),
        bench_regalloc(&xlarge, "xlarge (10fn)"),
        bench_regalloc(&stress, "stress (20fn)"),
        bench_regalloc(&xxh3, "xxh3 (loop)"),
    ];

    // --- E2E benchmarks (serial and parallel) ---
    eprintln!("  Benchmarking E2E (serial)...");
    let e2e_serial_small = bench_e2e(&small, "small (1fn)", false);
    let e2e_serial_medium = bench_e2e(&medium, "medium (3fn)", false);
    let e2e_serial_large = bench_e2e(&large, "large (5fn)", false);
    let e2e_serial_xlarge = bench_e2e(&xlarge, "xlarge (10fn)", false);
    let e2e_serial_stress = bench_e2e(&stress, "stress (20fn)", false);
    let e2e_serial_xxh3 = bench_e2e(&xxh3, "xxh3 (loop)", false);

    eprintln!("  Benchmarking E2E (parallel)...");
    let e2e_parallel_small = bench_e2e(&small, "small (1fn)", true);
    let e2e_parallel_medium = bench_e2e(&medium, "medium (3fn)", true);
    let e2e_parallel_large = bench_e2e(&large, "large (5fn)", true);
    let e2e_parallel_xlarge = bench_e2e(&xlarge, "xlarge (10fn)", true);
    let e2e_parallel_stress = bench_e2e(&stress, "stress (20fn)", true);
    let e2e_parallel_xxh3 = bench_e2e(&xxh3, "xxh3 (loop)", true);

    let e2e_results = vec![
        e2e_serial_small,
        e2e_serial_medium,
        e2e_serial_large,
        e2e_serial_xlarge,
        e2e_serial_stress,
        e2e_serial_xxh3,
        e2e_parallel_small,
        e2e_parallel_medium,
        e2e_parallel_large,
        e2e_parallel_xlarge,
        e2e_parallel_stress,
        e2e_parallel_xxh3,
    ];

    print_report(&isel_results, &opt_results, &regalloc_results, &e2e_results);
}
