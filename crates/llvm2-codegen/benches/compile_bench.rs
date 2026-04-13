// llvm2-codegen/benches/compile_bench.rs - Compilation throughput benchmarks
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Benchmark framework for LLVM2 compilation pipeline.
// Measures throughput (instructions/second) and per-phase timing.
//
// Run with: cargo bench -p llvm2-codegen --bench compile_bench
//
// Part of #65 — Benchmark suite

use std::time::{Duration, Instant};

use llvm2_codegen::pipeline::{
    build_add_test_function, encode_function, PipelineConfig, OptLevel,
};
use llvm2_codegen::macho::MachOWriter;
use llvm2_ir::function::{MachFunction, Signature, Type};
use llvm2_ir::inst::{AArch64Opcode, MachInst};
use llvm2_ir::operand::MachOperand;
use llvm2_ir::regs::{X0, X1, X8, X9};
use llvm2_ir::types::BlockId;

// ---------------------------------------------------------------------------
// Benchmark configuration
// ---------------------------------------------------------------------------

/// Number of warmup iterations before measurement.
const WARMUP_ITERS: u32 = 5;

/// Number of measurement iterations.
const BENCH_ITERS: u32 = 100;

/// Minimum benchmark duration per test case (ensures stable timings).
const MIN_BENCH_DURATION: Duration = Duration::from_millis(500);

// ---------------------------------------------------------------------------
// Test function builders — varying complexity levels
// ---------------------------------------------------------------------------

/// Complexity level for benchmark functions.
#[derive(Debug, Clone, Copy)]
enum Complexity {
    /// Trivial: 2 instructions (MOVZ + RET).
    Trivial,
    /// Simple: 3 instructions (ADD + RET with setup).
    Simple,
    /// Medium: ~10 instructions (arithmetic chain with multiple operations).
    Medium,
    /// Complex: ~30 instructions (loop with branches, multiple blocks).
    Complex,
    /// Large: ~100 instructions (unrolled computation).
    Large,
}

impl Complexity {
    fn name(&self) -> &'static str {
        match self {
            Complexity::Trivial => "trivial_2inst",
            Complexity::Simple => "simple_3inst",
            Complexity::Medium => "medium_10inst",
            Complexity::Complex => "complex_30inst",
            Complexity::Large => "large_100inst",
        }
    }

    #[allow(dead_code)]
    fn expected_inst_count(&self) -> usize {
        match self {
            Complexity::Trivial => 2,
            Complexity::Simple => 3,
            Complexity::Medium => 10,
            Complexity::Complex => 30,
            Complexity::Large => 100,
        }
    }
}

/// Build `return_const() -> i32`: MOVZ X0, #42; RET
fn build_trivial() -> MachFunction {
    let sig = Signature::new(vec![], vec![Type::I32]);
    let mut func = MachFunction::new("bench_trivial".to_string(), sig);
    let entry = func.entry;

    let mov = MachInst::new(
        AArch64Opcode::Movz,
        vec![MachOperand::PReg(X0), MachOperand::Imm(42)],
    );
    let mov_id = func.push_inst(mov);
    func.append_inst(entry, mov_id);

    let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
    let ret_id = func.push_inst(ret);
    func.append_inst(entry, ret_id);

    func
}

/// Build `add(a, b) -> i32`: ADD X0, X0, X1; RET
fn build_simple() -> MachFunction {
    build_add_test_function()
}

/// Build a medium-complexity function: chain of arithmetic (~10 instructions).
///
/// Computes: result = ((a + b) - 1) + ((a - b) + 2)
/// Uses X0, X1 as inputs, X8, X9 as temporaries.
fn build_medium() -> MachFunction {
    let sig = Signature::new(
        vec![Type::I64, Type::I64],
        vec![Type::I64],
    );
    let mut func = MachFunction::new("bench_medium".to_string(), sig);
    let entry = func.entry;

    // X8 = X0 + X1
    append_inst(&mut func, entry, AArch64Opcode::AddRR,
        vec![MachOperand::PReg(X8), MachOperand::PReg(X0), MachOperand::PReg(X1)]);
    // X8 = X8 - 1
    append_inst(&mut func, entry, AArch64Opcode::SubRI,
        vec![MachOperand::PReg(X8), MachOperand::PReg(X8), MachOperand::Imm(1)]);
    // X9 = X0 - X1
    append_inst(&mut func, entry, AArch64Opcode::SubRR,
        vec![MachOperand::PReg(X9), MachOperand::PReg(X0), MachOperand::PReg(X1)]);
    // X9 = X9 + 2
    append_inst(&mut func, entry, AArch64Opcode::AddRI,
        vec![MachOperand::PReg(X9), MachOperand::PReg(X9), MachOperand::Imm(2)]);
    // X0 = X8 + X9
    append_inst(&mut func, entry, AArch64Opcode::AddRR,
        vec![MachOperand::PReg(X0), MachOperand::PReg(X8), MachOperand::PReg(X9)]);
    // X0 = X0 + 10
    append_inst(&mut func, entry, AArch64Opcode::AddRI,
        vec![MachOperand::PReg(X0), MachOperand::PReg(X0), MachOperand::Imm(10)]);
    // X8 = X0 - X1
    append_inst(&mut func, entry, AArch64Opcode::SubRR,
        vec![MachOperand::PReg(X8), MachOperand::PReg(X0), MachOperand::PReg(X1)]);
    // X0 = X8 + X0
    append_inst(&mut func, entry, AArch64Opcode::AddRR,
        vec![MachOperand::PReg(X0), MachOperand::PReg(X8), MachOperand::PReg(X0)]);
    // X0 = X0 - 5
    append_inst(&mut func, entry, AArch64Opcode::SubRI,
        vec![MachOperand::PReg(X0), MachOperand::PReg(X0), MachOperand::Imm(5)]);
    // RET
    append_inst(&mut func, entry, AArch64Opcode::Ret, vec![]);

    func
}

/// Build a complex function: conditional branches with multiple blocks (~30 instructions).
///
/// Implements a max-of-3 with additional arithmetic:
///   Block 0: setup, CMP, B.GT
///   Block 1: fallthrough path
///   Block 2: taken path
///   Block 3: merge, more arithmetic, RET
fn build_complex() -> MachFunction {
    let sig = Signature::new(
        vec![Type::I64, Type::I64],
        vec![Type::I64],
    );
    let mut func = MachFunction::new("bench_complex".to_string(), sig);

    let bb0 = func.entry;
    let bb1 = func.create_block();
    let bb2 = func.create_block();
    let bb3 = func.create_block();

    // bb0: setup + first comparison
    append_inst(&mut func, bb0, AArch64Opcode::AddRI,
        vec![MachOperand::PReg(X8), MachOperand::PReg(X0), MachOperand::Imm(1)]);
    append_inst(&mut func, bb0, AArch64Opcode::SubRI,
        vec![MachOperand::PReg(X9), MachOperand::PReg(X1), MachOperand::Imm(1)]);
    append_inst(&mut func, bb0, AArch64Opcode::CmpRR,
        vec![MachOperand::PReg(X8), MachOperand::PReg(X9)]);
    append_inst(&mut func, bb0, AArch64Opcode::BCond,
        vec![MachOperand::Imm(0xC), MachOperand::Imm(6)]); // B.GT bb2 (skip 6 instrs)

    // bb1: fallthrough — X0 = complex arithmetic chain
    append_inst(&mut func, bb1, AArch64Opcode::AddRR,
        vec![MachOperand::PReg(X0), MachOperand::PReg(X0), MachOperand::PReg(X1)]);
    append_inst(&mut func, bb1, AArch64Opcode::SubRI,
        vec![MachOperand::PReg(X0), MachOperand::PReg(X0), MachOperand::Imm(3)]);
    append_inst(&mut func, bb1, AArch64Opcode::AddRI,
        vec![MachOperand::PReg(X8), MachOperand::PReg(X0), MachOperand::Imm(7)]);
    append_inst(&mut func, bb1, AArch64Opcode::SubRR,
        vec![MachOperand::PReg(X0), MachOperand::PReg(X8), MachOperand::PReg(X1)]);
    append_inst(&mut func, bb1, AArch64Opcode::AddRI,
        vec![MachOperand::PReg(X0), MachOperand::PReg(X0), MachOperand::Imm(2)]);
    append_inst(&mut func, bb1, AArch64Opcode::B,
        vec![MachOperand::Imm(6)]); // B bb3 (skip bb2's 5 instrs + this = 6)

    // bb2: taken path — X0 = different chain
    append_inst(&mut func, bb2, AArch64Opcode::SubRR,
        vec![MachOperand::PReg(X0), MachOperand::PReg(X0), MachOperand::PReg(X1)]);
    append_inst(&mut func, bb2, AArch64Opcode::AddRI,
        vec![MachOperand::PReg(X0), MachOperand::PReg(X0), MachOperand::Imm(10)]);
    append_inst(&mut func, bb2, AArch64Opcode::SubRI,
        vec![MachOperand::PReg(X8), MachOperand::PReg(X0), MachOperand::Imm(1)]);
    append_inst(&mut func, bb2, AArch64Opcode::AddRR,
        vec![MachOperand::PReg(X0), MachOperand::PReg(X8), MachOperand::PReg(X0)]);
    append_inst(&mut func, bb2, AArch64Opcode::SubRI,
        vec![MachOperand::PReg(X0), MachOperand::PReg(X0), MachOperand::Imm(4)]);

    // bb3: merge + final arithmetic + return
    append_inst(&mut func, bb3, AArch64Opcode::AddRR,
        vec![MachOperand::PReg(X8), MachOperand::PReg(X0), MachOperand::PReg(X0)]);
    append_inst(&mut func, bb3, AArch64Opcode::SubRI,
        vec![MachOperand::PReg(X8), MachOperand::PReg(X8), MachOperand::Imm(1)]);
    append_inst(&mut func, bb3, AArch64Opcode::AddRR,
        vec![MachOperand::PReg(X0), MachOperand::PReg(X8), MachOperand::PReg(X1)]);
    append_inst(&mut func, bb3, AArch64Opcode::SubRI,
        vec![MachOperand::PReg(X0), MachOperand::PReg(X0), MachOperand::Imm(2)]);
    append_inst(&mut func, bb3, AArch64Opcode::AddRI,
        vec![MachOperand::PReg(X8), MachOperand::PReg(X0), MachOperand::Imm(5)]);
    append_inst(&mut func, bb3, AArch64Opcode::SubRR,
        vec![MachOperand::PReg(X0), MachOperand::PReg(X8), MachOperand::PReg(X1)]);
    append_inst(&mut func, bb3, AArch64Opcode::AddRI,
        vec![MachOperand::PReg(X0), MachOperand::PReg(X0), MachOperand::Imm(1)]);
    append_inst(&mut func, bb3, AArch64Opcode::CmpRR,
        vec![MachOperand::PReg(X0), MachOperand::PReg(X8)]);
    // Second conditional pair inside merge block
    append_inst(&mut func, bb3, AArch64Opcode::SubRI,
        vec![MachOperand::PReg(X0), MachOperand::PReg(X0), MachOperand::Imm(1)]);
    append_inst(&mut func, bb3, AArch64Opcode::AddRR,
        vec![MachOperand::PReg(X0), MachOperand::PReg(X0), MachOperand::PReg(X1)]);
    append_inst(&mut func, bb3, AArch64Opcode::Ret, vec![]);

    func
}

/// Build a large function: ~100 instructions of unrolled arithmetic.
///
/// Simulates a hot loop body that has been fully unrolled, using only
/// ADD/SUB register-register and register-immediate operations.
fn build_large() -> MachFunction {
    let sig = Signature::new(
        vec![Type::I64, Type::I64],
        vec![Type::I64],
    );
    let mut func = MachFunction::new("bench_large".to_string(), sig);
    let entry = func.entry;

    // Generate ~99 arithmetic instructions + RET
    let regs = [X0, X1, X8, X9];
    let opcodes_rr = [AArch64Opcode::AddRR, AArch64Opcode::SubRR];
    let opcodes_ri = [AArch64Opcode::AddRI, AArch64Opcode::SubRI];

    for i in 0..99u32 {
        let dst = regs[(i % 4) as usize];
        let src1 = regs[((i + 1) % 4) as usize];

        if i % 3 == 0 {
            // Register-immediate
            let op = &opcodes_ri[(i as usize / 3) % 2];
            append_inst(&mut func, entry, op.clone(),
                vec![
                    MachOperand::PReg(dst),
                    MachOperand::PReg(src1),
                    MachOperand::Imm((i as i64 % 100) + 1),
                ]);
        } else {
            // Register-register
            let src2 = regs[((i + 2) % 4) as usize];
            let op = &opcodes_rr[(i as usize) % 2];
            append_inst(&mut func, entry, op.clone(),
                vec![
                    MachOperand::PReg(dst),
                    MachOperand::PReg(src1),
                    MachOperand::PReg(src2),
                ]);
        }
    }

    // Final: MOV result to X0 and RET
    append_inst(&mut func, entry, AArch64Opcode::Ret, vec![]);

    func
}

/// Helper: append an instruction to a block.
fn append_inst(
    func: &mut MachFunction,
    block: BlockId,
    opcode: AArch64Opcode,
    operands: Vec<MachOperand>,
) {
    let inst = MachInst::new(opcode, operands);
    let id = func.push_inst(inst);
    func.append_inst(block, id);
}

/// Build a function for the given complexity level.
fn build_function(complexity: Complexity) -> MachFunction {
    match complexity {
        Complexity::Trivial => build_trivial(),
        Complexity::Simple => build_simple(),
        Complexity::Medium => build_medium(),
        Complexity::Complex => build_complex(),
        Complexity::Large => build_large(),
    }
}

// ---------------------------------------------------------------------------
// Phase-level timing
// ---------------------------------------------------------------------------

/// Results from timing a single pipeline phase.
#[derive(Debug, Clone)]
struct PhaseResult {
    name: &'static str,
    duration: Duration,
}

/// Results from a complete benchmark run.
#[derive(Debug)]
struct BenchResult {
    complexity: &'static str,
    inst_count: usize,
    total_duration: Duration,
    phases: Vec<PhaseResult>,
    iterations: u32,
}

impl BenchResult {
    fn instructions_per_second(&self) -> f64 {
        let total_secs = self.total_duration.as_secs_f64();
        if total_secs == 0.0 {
            return f64::INFINITY;
        }
        (self.inst_count as f64 * self.iterations as f64) / total_secs
    }

    #[allow(dead_code)]
    fn avg_per_function_us(&self) -> f64 {
        self.total_duration.as_micros() as f64 / self.iterations as f64
    }
}

/// Time the optimization phase in isolation.
fn time_optimization(func: &MachFunction, config: &PipelineConfig, iters: u32) -> PhaseResult {
    use llvm2_opt::pipeline::{OptLevel as OptOptLevel, OptimizationPipeline};

    let opt_level = match config.opt_level {
        OptLevel::O0 => OptOptLevel::O0,
        OptLevel::O1 => OptOptLevel::O1,
        OptLevel::O2 => OptOptLevel::O2,
        OptLevel::O3 => OptOptLevel::O3,
    };

    let start = Instant::now();
    for _ in 0..iters {
        let mut f = func.clone();
        let pipeline = OptimizationPipeline::new(opt_level);
        let _stats = pipeline.run(&mut f);
    }
    PhaseResult {
        name: "Optimization",
        duration: start.elapsed(),
    }
}

/// Time the encoding phase in isolation.
fn time_encoding(func: &MachFunction, iters: u32) -> PhaseResult {
    let start = Instant::now();
    for _ in 0..iters {
        let _ = encode_function(func);
    }
    PhaseResult {
        name: "Encoding",
        duration: start.elapsed(),
    }
}

/// Time Mach-O emission in isolation.
fn time_emit(func_name: &str, code: &[u8], iters: u32) -> PhaseResult {
    let start = Instant::now();
    for _ in 0..iters {
        let mut writer = MachOWriter::new();
        writer.add_text_section(code);
        let symbol_name = format!("_{}", func_name);
        writer.add_symbol(&symbol_name, 1, 0, true);
        let _ = writer.write();
    }
    PhaseResult {
        name: "Mach-O Emit",
        duration: start.elapsed(),
    }
}

/// Time frame lowering in isolation.
fn time_frame_lowering(func: &MachFunction, iters: u32) -> PhaseResult {
    use llvm2_codegen::frame;

    let start = Instant::now();
    for _ in 0..iters {
        let mut f = func.clone();
        let layout = frame::compute_frame_layout(&f, 0, true);
        frame::eliminate_frame_indices(&mut f, &layout);
        frame::insert_prologue_epilogue(&mut f, &layout);
    }
    PhaseResult {
        name: "Frame Lower",
        duration: start.elapsed(),
    }
}

/// Run a complete benchmark for one complexity level.
fn bench_complexity(complexity: Complexity, opt_level: OptLevel) -> BenchResult {
    let func = build_function(complexity);
    let inst_count = func.insts.len();
    let config = PipelineConfig {
        opt_level,
        emit_debug: false,
    };

    // Warmup
    for _ in 0..WARMUP_ITERS {
        let f = func.clone();
        let _ = encode_function(&f);
    }

    // Determine iteration count: at least BENCH_ITERS, but enough to fill MIN_BENCH_DURATION
    let probe_start = Instant::now();
    for _ in 0..10 {
        let f = func.clone();
        let _ = encode_function(&f);
    }
    let probe_elapsed = probe_start.elapsed();
    let per_iter_ns = probe_elapsed.as_nanos() / 10;
    let min_iters = if per_iter_ns > 0 {
        (MIN_BENCH_DURATION.as_nanos() / per_iter_ns) as u32
    } else {
        BENCH_ITERS
    };
    let iters = min_iters.max(BENCH_ITERS);

    // Phase-level measurements
    let opt_result = time_optimization(&func, &config, iters);
    let frame_result = time_frame_lowering(&func, iters);
    let encode_result = time_encoding(&func, iters);
    let code = encode_function(&func).unwrap_or_default();
    let emit_result = time_emit(&func.name, &code, iters);

    // Total pipeline measurement (all phases together)
    let total_start = Instant::now();
    for _ in 0..iters {
        let mut f = func.clone();
        // Optimization
        {
            use llvm2_opt::pipeline::{OptLevel as OptOptLevel, OptimizationPipeline};
            let opt_level = match config.opt_level {
                OptLevel::O0 => OptOptLevel::O0,
                OptLevel::O1 => OptOptLevel::O1,
                OptLevel::O2 => OptOptLevel::O2,
                OptLevel::O3 => OptOptLevel::O3,
            };
            let pipeline = OptimizationPipeline::new(opt_level);
            let _stats = pipeline.run(&mut f);
        }
        // Frame lowering
        {
            use llvm2_codegen::frame;
            let layout = frame::compute_frame_layout(&f, 0, true);
            frame::eliminate_frame_indices(&mut f, &layout);
            frame::insert_prologue_epilogue(&mut f, &layout);
        }
        // Encoding
        let code = encode_function(&f).unwrap_or_default();
        // Mach-O emission
        {
            let mut writer = MachOWriter::new();
            writer.add_text_section(&code);
            let symbol_name = format!("_{}", f.name);
            writer.add_symbol(&symbol_name, 1, 0, true);
            let _ = writer.write();
        }
    }
    let total_duration = total_start.elapsed();

    BenchResult {
        complexity: complexity.name(),
        inst_count,
        total_duration,
        phases: vec![opt_result, frame_result, encode_result, emit_result],
        iterations: iters,
    }
}

// ---------------------------------------------------------------------------
// Human-readable report
// ---------------------------------------------------------------------------

fn format_duration(d: Duration, iters: u32) -> String {
    let per_iter_ns = d.as_nanos() / iters as u128;
    if per_iter_ns < 1_000 {
        format!("{:>6} ns", per_iter_ns)
    } else if per_iter_ns < 1_000_000 {
        format!("{:>6.1} us", per_iter_ns as f64 / 1_000.0)
    } else {
        format!("{:>6.1} ms", per_iter_ns as f64 / 1_000_000.0)
    }
}

fn print_report(results: &[BenchResult]) {
    println!();
    println!("==============================================================================");
    println!("  LLVM2 Compilation Benchmark Report");
    println!("==============================================================================");
    println!();
    println!("  Optimization level: O0 (no optimizations, measures raw pipeline overhead)");
    println!("  Architecture: AArch64 (Apple Silicon)");
    println!("  Note: ISel and RegAlloc are not benchmarked (require tMIR input / VRegs)");
    println!("  Phases measured: Optimization, Frame Lowering, Encoding, Mach-O Emission");
    println!();

    // Summary table
    println!("  {:<20} {:>6} {:>12} {:>14} {:>12}",
        "Function", "Insts", "Time/func", "Insts/sec", "Iterations");
    println!("  {:-<20} {:->6} {:->12} {:->14} {:->12}",
        "", "", "", "", "");

    for r in results {
        println!("  {:<20} {:>6} {:>12} {:>14.0} {:>12}",
            r.complexity,
            r.inst_count,
            format_duration(r.total_duration, r.iterations),
            r.instructions_per_second(),
            r.iterations);
    }

    // Per-phase breakdown
    println!();
    println!("  Per-Phase Breakdown (average per function):");
    println!();
    println!("  {:<20} {:>12} {:>12} {:>12} {:>12}",
        "Function", "Optimize", "FrameLower", "Encode", "Mach-O");
    println!("  {:-<20} {:->12} {:->12} {:->12} {:->12}",
        "", "", "", "", "");

    for r in results {
        let phase_strs: Vec<String> = r.phases.iter()
            .map(|p| format_duration(p.duration, r.iterations))
            .collect();
        println!("  {:<20} {:>12} {:>12} {:>12} {:>12}",
            r.complexity,
            phase_strs.get(0).map(|s| s.as_str()).unwrap_or("N/A"),
            phase_strs.get(1).map(|s| s.as_str()).unwrap_or("N/A"),
            phase_strs.get(2).map(|s| s.as_str()).unwrap_or("N/A"),
            phase_strs.get(3).map(|s| s.as_str()).unwrap_or("N/A"));
    }

    // Phase percentage breakdown for largest function
    if let Some(largest) = results.last() {
        let total_phase_ns: u128 = largest.phases.iter()
            .map(|p| p.duration.as_nanos())
            .sum();
        if total_phase_ns > 0 {
            println!();
            println!("  Phase time distribution ({}):  ", largest.complexity);
            for phase in &largest.phases {
                let pct = phase.duration.as_nanos() as f64 / total_phase_ns as f64 * 100.0;
                let bar_len = (pct / 2.0) as usize;
                let bar: String = "#".repeat(bar_len);
                println!("    {:<14} {:>5.1}%  {}", phase.name, pct, bar);
            }
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
    let complexities = [
        Complexity::Trivial,
        Complexity::Simple,
        Complexity::Medium,
        Complexity::Complex,
        Complexity::Large,
    ];

    let mut results = Vec::new();
    for &complexity in &complexities {
        eprint!("  Benchmarking {}...", complexity.name());
        let result = bench_complexity(complexity, OptLevel::O0);
        eprintln!(" done ({} iterations)", result.iterations);
        results.push(result);
    }

    print_report(&results);
}
