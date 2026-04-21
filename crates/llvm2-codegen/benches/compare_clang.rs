// llvm2-codegen/benches/compare_clang.rs - LLVM2 vs clang -O2 comparison
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Benchmark comparison: measures LLVM2 compile time and instruction count
// against clang -O2 for equivalent functions. LLVM2 operates on pre-built
// MachIR (post-ISel, post-RegAlloc), so we measure the encoding + Mach-O
// pipeline. clang compiles equivalent C source files from scratch.
//
// Run with: cargo bench -p llvm2-codegen --bench compare_clang
//
// Part of #65 — Benchmark suite

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use llvm2_codegen::macho::MachOWriter;
use llvm2_codegen::pipeline::encode_function;
use llvm2_ir::function::{MachFunction, Signature, Type};
use llvm2_ir::inst::{AArch64Opcode, MachInst};
use llvm2_ir::operand::MachOperand;
use llvm2_ir::regs::{X0, X1, X8, X9};
use llvm2_ir::types::BlockId;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Warmup iterations for LLVM2 benchmarks.
const WARMUP_ITERS: u32 = 10;

/// Minimum measurement iterations for LLVM2 benchmarks.
const BENCH_ITERS: u32 = 200;

/// Minimum benchmark duration to ensure stable timings.
const MIN_BENCH_DURATION: Duration = Duration::from_millis(500);

/// Number of clang invocations per test (clang is slow, fewer iters needed).
const CLANG_ITERS: u32 = 5;

// ---------------------------------------------------------------------------
// Test function descriptors
// ---------------------------------------------------------------------------

/// A test case: pairs a name with an LLVM2 MachFunction builder and a C source file.
struct TestCase {
    name: &'static str,
    description: &'static str,
    c_filename: &'static str,
    build_fn: fn() -> MachFunction,
}

fn test_cases() -> Vec<TestCase> {
    vec![
        TestCase {
            name: "trivial",
            description: "return 42",
            c_filename: "trivial.c",
            build_fn: build_trivial,
        },
        TestCase {
            name: "add",
            description: "a + b",
            c_filename: "add.c",
            build_fn: build_add,
        },
        TestCase {
            name: "max",
            description: "if a > b { a } else { b }",
            c_filename: "max.c",
            build_fn: build_max,
        },
        TestCase {
            name: "factorial",
            description: "loop with multiply",
            c_filename: "factorial.c",
            build_fn: build_factorial,
        },
        TestCase {
            name: "fibonacci",
            description: "loop, two accumulators",
            c_filename: "fibonacci.c",
            build_fn: build_fibonacci,
        },
    ]
}

// ---------------------------------------------------------------------------
// LLVM2 test function builders (pre-allocated physical registers)
// ---------------------------------------------------------------------------

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

/// `trivial() -> i64`: MOVZ X0, #42; RET
fn build_trivial() -> MachFunction {
    let sig = Signature::new(vec![], vec![Type::I64]);
    let mut func = MachFunction::new("trivial".to_string(), sig);
    let entry = func.entry;

    append_inst(&mut func, entry, AArch64Opcode::Movz,
        vec![MachOperand::PReg(X0), MachOperand::Imm(42)]);
    append_inst(&mut func, entry, AArch64Opcode::Ret, vec![]);

    func
}

/// `add(a: i64, b: i64) -> i64`: ADD X0, X0, X1; RET
fn build_add() -> MachFunction {
    let sig = Signature::new(vec![Type::I64, Type::I64], vec![Type::I64]);
    let mut func = MachFunction::new("add".to_string(), sig);
    let entry = func.entry;

    append_inst(&mut func, entry, AArch64Opcode::AddRR,
        vec![MachOperand::PReg(X0), MachOperand::PReg(X0), MachOperand::PReg(X1)]);
    append_inst(&mut func, entry, AArch64Opcode::Ret, vec![]);

    func
}

/// `max(a: i64, b: i64) -> i64`:
///   CMP X0, X1; B.GT taken; MOV X0, X1; RET; taken: RET
///
/// This implements: if (a > b) return a; else return b;
/// clang -O2 would emit CSEL, but our IR models this with branches.
fn build_max() -> MachFunction {
    let sig = Signature::new(vec![Type::I64, Type::I64], vec![Type::I64]);
    let mut func = MachFunction::new("max".to_string(), sig);

    let bb0 = func.entry;
    let bb1 = func.create_block();  // else: MOV X0, X1; RET
    let bb2 = func.create_block();  // then: RET (X0 already has a)

    // bb0: CMP X0, X1; B.GT bb2 (skip bb1)
    append_inst(&mut func, bb0, AArch64Opcode::CmpRR,
        vec![MachOperand::PReg(X0), MachOperand::PReg(X1)]);
    // B.GT: condition code 0xC = GT, offset = 2 instructions forward (skip bb1)
    append_inst(&mut func, bb0, AArch64Opcode::BCond,
        vec![MachOperand::Imm(0xC), MachOperand::Imm(3)]);

    // bb1 (else): X0 = X1, then RET
    append_inst(&mut func, bb1, AArch64Opcode::MovR,
        vec![MachOperand::PReg(X0), MachOperand::PReg(X1)]);
    append_inst(&mut func, bb1, AArch64Opcode::Ret, vec![]);

    // bb2 (then): a is already in X0, just RET
    append_inst(&mut func, bb2, AArch64Opcode::Ret, vec![]);

    func
}

/// `factorial(n: i64) -> i64`:
///   X8 = 1 (result); X9 = 2 (i)
///   loop: CMP X9, X0; B.GT done
///         MUL X8, X8, X9; ADD X9, X9, #1; B loop
///   done: MOV X0, X8; RET
fn build_factorial() -> MachFunction {
    let sig = Signature::new(vec![Type::I64], vec![Type::I64]);
    let mut func = MachFunction::new("factorial".to_string(), sig);

    let entry = func.entry;
    let loop_bb = func.create_block();
    let done = func.create_block();

    // entry: X8 = 1 (result), X9 = 2 (i), B loop
    append_inst(&mut func, entry, AArch64Opcode::Movz,
        vec![MachOperand::PReg(X8), MachOperand::Imm(1)]);
    append_inst(&mut func, entry, AArch64Opcode::Movz,
        vec![MachOperand::PReg(X9), MachOperand::Imm(2)]);
    append_inst(&mut func, entry, AArch64Opcode::B,
        vec![MachOperand::Imm(1)]);  // skip 0 to loop header

    // loop: CMP X9, X0; B.GT done; MUL X8, X8, X9; ADD X9, X9, #1; B loop
    append_inst(&mut func, loop_bb, AArch64Opcode::CmpRR,
        vec![MachOperand::PReg(X9), MachOperand::PReg(X0)]);
    append_inst(&mut func, loop_bb, AArch64Opcode::BCond,
        vec![MachOperand::Imm(0xC), MachOperand::Imm(4)]);  // B.GT done (skip 4 instrs)
    append_inst(&mut func, loop_bb, AArch64Opcode::MulRR,
        vec![MachOperand::PReg(X8), MachOperand::PReg(X8), MachOperand::PReg(X9)]);
    append_inst(&mut func, loop_bb, AArch64Opcode::AddRI,
        vec![MachOperand::PReg(X9), MachOperand::PReg(X9), MachOperand::Imm(1)]);
    append_inst(&mut func, loop_bb, AArch64Opcode::B,
        vec![MachOperand::Imm(-4)]);  // B loop (back 4 instructions)

    // done: MOV X0, X8; RET
    append_inst(&mut func, done, AArch64Opcode::MovR,
        vec![MachOperand::PReg(X0), MachOperand::PReg(X8)]);
    append_inst(&mut func, done, AArch64Opcode::Ret, vec![]);

    func
}

/// `fibonacci(n: i64) -> i64`:
///   X8 = 0 (a); X9 = 1 (b); X1 = 0 (i)
///   loop: CMP X1, X0; B.GE done
///         X2 = X8 + X9; X8 = X9; X9 = X2; ADD X1, X1, #1; B loop
///   done: MOV X0, X8; RET
///
/// Uses X1 as loop counter (clobbers second arg, but n is in X0 and
/// we only need it for the comparison).
fn build_fibonacci() -> MachFunction {
    use llvm2_ir::regs::X2;

    let sig = Signature::new(vec![Type::I64], vec![Type::I64]);
    let mut func = MachFunction::new("fibonacci".to_string(), sig);

    let entry = func.entry;
    let loop_bb = func.create_block();
    let done = func.create_block();

    // entry: X8 = 0 (a), X9 = 1 (b), X1 = 0 (i), B loop
    append_inst(&mut func, entry, AArch64Opcode::Movz,
        vec![MachOperand::PReg(X8), MachOperand::Imm(0)]);
    append_inst(&mut func, entry, AArch64Opcode::Movz,
        vec![MachOperand::PReg(X9), MachOperand::Imm(1)]);
    append_inst(&mut func, entry, AArch64Opcode::Movz,
        vec![MachOperand::PReg(X1), MachOperand::Imm(0)]);
    append_inst(&mut func, entry, AArch64Opcode::B,
        vec![MachOperand::Imm(1)]);  // B loop

    // loop: CMP X1, X0; B.GE done; X2=X8+X9; X8=X9; X9=X2; X1+=1; B loop
    append_inst(&mut func, loop_bb, AArch64Opcode::CmpRR,
        vec![MachOperand::PReg(X1), MachOperand::PReg(X0)]);
    append_inst(&mut func, loop_bb, AArch64Opcode::BCond,
        vec![MachOperand::Imm(0xA), MachOperand::Imm(6)]);  // B.GE done (0xA = GE, skip 6)
    append_inst(&mut func, loop_bb, AArch64Opcode::AddRR,
        vec![MachOperand::PReg(X2), MachOperand::PReg(X8), MachOperand::PReg(X9)]);
    append_inst(&mut func, loop_bb, AArch64Opcode::MovR,
        vec![MachOperand::PReg(X8), MachOperand::PReg(X9)]);
    append_inst(&mut func, loop_bb, AArch64Opcode::MovR,
        vec![MachOperand::PReg(X9), MachOperand::PReg(X2)]);
    append_inst(&mut func, loop_bb, AArch64Opcode::AddRI,
        vec![MachOperand::PReg(X1), MachOperand::PReg(X1), MachOperand::Imm(1)]);
    append_inst(&mut func, loop_bb, AArch64Opcode::B,
        vec![MachOperand::Imm(-6)]);  // B loop (back 6)

    // done: MOV X0, X8; RET
    append_inst(&mut func, done, AArch64Opcode::MovR,
        vec![MachOperand::PReg(X0), MachOperand::PReg(X8)]);
    append_inst(&mut func, done, AArch64Opcode::Ret, vec![]);

    func
}

// ---------------------------------------------------------------------------
// LLVM2 compilation: encode + Mach-O
// ---------------------------------------------------------------------------

/// Compile a MachFunction through LLVM2's backend: optimize + frame lower +
/// encode + Mach-O emission. Returns (object bytes, instruction count).
fn llvm2_compile(func: &MachFunction) -> (Vec<u8>, usize) {
    let mut f = func.clone();

    // Run optimization (O0 for fair comparison — we're measuring pipeline overhead)
    {
        use llvm2_opt::pipeline::{OptLevel as OptOptLevel, OptimizationPipeline};
        let pipeline = OptimizationPipeline::new(OptOptLevel::O0);
        let _stats = pipeline.run(&mut f);
    }

    // Frame lowering
    {
        use llvm2_codegen::frame;
        let layout = frame::compute_frame_layout(&f, 0, true);
        frame::eliminate_frame_indices(&mut f, &layout);
        frame::insert_prologue_epilogue(&mut f, &layout);
    }

    // Count instructions before encoding (non-pseudo only)
    let inst_count = count_instructions(&f);

    // Encode
    let code = encode_function(&f).unwrap_or_else(|e| {
        eprintln!("  WARNING: encoding failed for {}: {}", f.name, e);
        Vec::new()
    });

    // Mach-O emission
    let mut writer = MachOWriter::new();
    writer.add_text_section(&code);
    let symbol_name = format!("_{}", f.name);
    writer.add_symbol(&symbol_name, 1, 0, true);
    let obj_bytes = writer.write();

    (obj_bytes, inst_count)
}

/// Count non-pseudo instructions in a MachFunction.
fn count_instructions(func: &MachFunction) -> usize {
    let mut count = 0;
    for &block_id in &func.block_order {
        let block = func.block(block_id);
        for &inst_id in &block.insts {
            let inst = func.inst(inst_id);
            if !inst.is_pseudo() {
                count += 1;
            }
        }
    }
    count
}

// ---------------------------------------------------------------------------
// clang compilation
// ---------------------------------------------------------------------------

/// Find the clang binary. Returns None if clang is not available.
fn find_clang() -> Option<PathBuf> {
    // Try common locations
    for candidate in &["clang", "/usr/bin/clang", "/opt/homebrew/opt/llvm/bin/clang"] {
        if let Ok(output) = Command::new(candidate)
            .arg("--version")
            .output()
        {
            if output.status.success() {
                return Some(PathBuf::from(candidate));
            }
        }
    }
    None
}

/// Get the clang version string (first line of --version output).
fn clang_version(clang: &Path) -> String {
    Command::new(clang)
        .arg("--version")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.lines().next().map(|l| l.to_string()))
        .unwrap_or_else(|| "unknown".to_string())
}

/// Time clang -O2 compilation of a C source file.
/// Returns the average duration per invocation over `iters` runs.
fn time_clang(clang: &Path, c_file: &Path, iters: u32) -> Option<Duration> {
    if !c_file.exists() {
        eprintln!("  WARNING: C source file not found: {}", c_file.display());
        return None;
    }

    // Warmup: 2 runs
    for _ in 0..2 {
        let status = Command::new(clang)
            .args(["-O2", "-c", "-o", "/dev/null"])
            .arg(c_file)
            .output();
        if status.is_err() || !status.unwrap().status.success() {
            eprintln!("  WARNING: clang compilation failed for {}", c_file.display());
            return None;
        }
    }

    // Timed runs
    let start = Instant::now();
    for _ in 0..iters {
        let _ = Command::new(clang)
            .args(["-O2", "-c", "-o", "/dev/null"])
            .arg(c_file)
            .output();
    }
    let elapsed = start.elapsed();

    Some(elapsed / iters)
}

/// Count AArch64 instructions in clang -O2 -S output.
///
/// Parses the assembly output and counts lines that look like instructions
/// (not labels, directives, or comments).
fn clang_instruction_count(clang: &Path, c_file: &Path) -> Option<usize> {
    if !c_file.exists() {
        return None;
    }

    let output = Command::new(clang)
        .args(["-O2", "-S", "-o", "-"])
        .arg(c_file)
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let asm = String::from_utf8(output.stdout).ok()?;
    let count = count_asm_instructions(&asm);
    Some(count)
}

/// Count AArch64 assembly instructions in a .s file's text.
///
/// Heuristic: an instruction line starts with optional whitespace, then a
/// lowercase letter (mnemonic). We skip:
/// - Lines starting with '.' (directives)
/// - Lines ending with ':' (labels)
/// - Lines starting with '//' or ';' (comments)
/// - Empty lines
/// - Lines outside .text section
fn count_asm_instructions(asm: &str) -> usize {
    let mut in_text = false;
    let mut count = 0;

    for line in asm.lines() {
        let trimmed = line.trim();

        // Track section transitions
        if trimmed.starts_with(".section") || trimmed.starts_with(".text") {
            in_text = trimmed.contains("__text") || trimmed.contains(".text");
            continue;
        }

        if !in_text {
            // Also handle the implicit .text from .globl onwards in simple files
            if trimmed.starts_with(".globl") {
                in_text = true;
            }
            continue;
        }

        // Skip empty lines
        if trimmed.is_empty() {
            continue;
        }

        // Skip comments
        if trimmed.starts_with("//") || trimmed.starts_with(';') || trimmed.starts_with('#') {
            continue;
        }

        // Skip directives
        if trimmed.starts_with('.') {
            continue;
        }

        // Skip labels (lines ending with ':')
        if trimmed.ends_with(':') {
            continue;
        }

        // Skip lines that are clearly not instructions
        if trimmed.starts_with("Lfunc") || trimmed.starts_with("ltmp") {
            continue;
        }

        // What remains should be an instruction
        // Verify it starts with a letter (mnemonic)
        if trimmed.chars().next().map_or(false, |c| c.is_ascii_alphabetic()) {
            count += 1;
        }
    }

    count
}

// ---------------------------------------------------------------------------
// Benchmark results
// ---------------------------------------------------------------------------

#[allow(dead_code)]
struct CompareResult {
    name: &'static str,
    description: &'static str,
    llvm2_time: Duration,
    clang_time: Option<Duration>,
    llvm2_inst_count: usize,
    clang_inst_count: Option<usize>,
    llvm2_iters: u32,
}

impl CompareResult {
    fn time_ratio(&self) -> Option<f64> {
        self.clang_time.map(|ct| {
            let llvm2_ns = self.llvm2_time.as_nanos() as f64;
            let clang_ns = ct.as_nanos() as f64;
            if clang_ns == 0.0 { 0.0 } else { llvm2_ns / clang_ns }
        })
    }

    fn inst_ratio(&self) -> Option<f64> {
        self.clang_inst_count.map(|ci| {
            if ci == 0 { 0.0 } else { self.llvm2_inst_count as f64 / ci as f64 }
        })
    }
}

// ---------------------------------------------------------------------------
// Benchmark driver
// ---------------------------------------------------------------------------

/// Locate the compare_inputs/ directory relative to the benchmark binary.
fn find_inputs_dir() -> PathBuf {
    // Try relative to crate root (when run via cargo bench)
    let candidates = [
        PathBuf::from("crates/llvm2-codegen/benches/compare_inputs"),
        PathBuf::from("benches/compare_inputs"),
        // Fallback: relative to CARGO_MANIFEST_DIR
        PathBuf::from(
            std::env::var("CARGO_MANIFEST_DIR")
                .unwrap_or_else(|_| ".".to_string())
        ).join("benches/compare_inputs"),
    ];

    for candidate in &candidates {
        if candidate.exists() {
            return candidate.clone();
        }
    }

    // Last resort: use the first candidate path (will produce warnings later)
    candidates[0].clone()
}

/// Run the LLVM2 benchmark for a single test case.
/// Returns (average duration, instruction count, iterations used).
fn bench_llvm2(func: &MachFunction) -> (Duration, usize, u32) {
    // Warmup
    for _ in 0..WARMUP_ITERS {
        let _ = llvm2_compile(func);
    }

    // Probe to determine iteration count
    let probe_start = Instant::now();
    for _ in 0..10 {
        let _ = llvm2_compile(func);
    }
    let probe_elapsed = probe_start.elapsed();
    let per_iter_ns = probe_elapsed.as_nanos() / 10;
    let min_iters = if per_iter_ns > 0 {
        (MIN_BENCH_DURATION.as_nanos() / per_iter_ns) as u32
    } else {
        BENCH_ITERS
    };
    let iters = min_iters.max(BENCH_ITERS);

    // Get instruction count from a single compile
    let (_, inst_count) = llvm2_compile(func);

    // Timed run
    let start = Instant::now();
    for _ in 0..iters {
        let _ = llvm2_compile(func);
    }
    let total = start.elapsed();
    let avg = total / iters;

    (avg, inst_count, iters)
}

// ---------------------------------------------------------------------------
// Report formatting
// ---------------------------------------------------------------------------

fn format_duration_human(d: Duration) -> String {
    let ns = d.as_nanos();
    if ns < 1_000 {
        format!("{} ns", ns)
    } else if ns < 1_000_000 {
        format!("{:.1} us", ns as f64 / 1_000.0)
    } else if ns < 1_000_000_000 {
        format!("{:.2} ms", ns as f64 / 1_000_000.0)
    } else {
        format!("{:.2} s", ns as f64 / 1_000_000_000.0)
    }
}

fn print_markdown_report(results: &[CompareResult], clang_version: &str, has_clang: bool) {
    println!();
    println!("# LLVM2 vs clang -O2 Comparison");
    println!();
    println!("- **Architecture:** AArch64 (Apple Silicon)");
    if has_clang {
        println!("- **clang:** {}", clang_version);
    } else {
        println!("- **clang:** NOT AVAILABLE (clang columns will be empty)");
    }
    println!("- **LLVM2 pipeline:** Optimize(O0) + FrameLower + Encode + Mach-O");
    println!("- **Note:** LLVM2 starts from pre-built MachIR; clang starts from C source.");
    println!("  This is intentional: LLVM2 is designed for the use case where ISel has");
    println!("  already produced MachIR. clang's time includes parsing, LLVM IR gen,");
    println!("  optimization, ISel, regalloc, encoding, AND process startup overhead.");
    println!();

    // Compile-time table
    println!("## Compile Time");
    println!();
    println!("| Function   | Description              | LLVM2 time | clang -O2 time | Ratio   | Speedup |");
    println!("|------------|--------------------------|------------|----------------|---------|---------|");

    for r in results {
        let llvm2_str = format_duration_human(r.llvm2_time);
        let clang_str = r.clang_time
            .map(|d| format_duration_human(d))
            .unwrap_or_else(|| "N/A".to_string());
        let ratio_str = r.time_ratio()
            .map(|r_val| format!("{:.4}x", r_val))
            .unwrap_or_else(|| "N/A".to_string());
        let speedup_str = r.time_ratio()
            .map(|r_val| if r_val > 0.0 { format!("{:.0}x", 1.0 / r_val) } else { "inf".to_string() })
            .unwrap_or_else(|| "N/A".to_string());

        println!("| {:<10} | {:<24} | {:>10} | {:>14} | {:>7} | {:>7} |",
            r.name, r.description, llvm2_str, clang_str, ratio_str, speedup_str);
    }

    // Instruction count table
    println!();
    println!("## Instruction Count (Code Quality)");
    println!();
    println!("| Function   | LLVM2 insts | clang insts | Ratio  |");
    println!("|------------|-------------|-------------|--------|");

    for r in results {
        let clang_str = r.clang_inst_count
            .map(|c| format!("{}", c))
            .unwrap_or_else(|| "N/A".to_string());
        let ratio_str = r.inst_ratio()
            .map(|r_val| format!("{:.2}x", r_val))
            .unwrap_or_else(|| "N/A".to_string());

        println!("| {:<10} | {:>11} | {:>11} | {:>6} |",
            r.name, r.llvm2_inst_count, clang_str, ratio_str);
    }

    // Summary
    println!();
    println!("## Summary");
    println!();

    let time_ratios: Vec<f64> = results.iter().filter_map(|r| r.time_ratio()).collect();
    if !time_ratios.is_empty() {
        let avg_ratio: f64 = time_ratios.iter().sum::<f64>() / time_ratios.len() as f64;
        let avg_speedup = if avg_ratio > 0.0 { 1.0 / avg_ratio } else { f64::INFINITY };
        println!("- **Average compile-time ratio:** {:.4}x (LLVM2 is ~{:.0}x faster)", avg_ratio, avg_speedup);
        println!("- LLVM2 operates on pre-built MachIR, avoiding C parsing, LLVM IR generation,");
        println!("  and LLVM's multi-pass optimization pipeline.");
        println!("- clang's time includes process startup (~10ms baseline on macOS).");
    }

    let inst_ratios: Vec<f64> = results.iter().filter_map(|r| r.inst_ratio()).collect();
    if !inst_ratios.is_empty() {
        let avg_inst_ratio: f64 = inst_ratios.iter().sum::<f64>() / inst_ratios.len() as f64;
        println!("- **Average instruction count ratio:** {:.2}x", avg_inst_ratio);
        if avg_inst_ratio > 1.5 {
            println!("- LLVM2 currently produces more instructions than clang -O2.");
            println!("  This is expected: LLVM2 uses conservative codegen (no CSEL, no peephole");
            println!("  opts at the MachIR level yet). Instruction count parity is a future goal.");
        } else {
            println!("- LLVM2 instruction counts are competitive with clang -O2.");
        }
    }

    println!();
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    // Support --validate flag for quick smoke test
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--validate") {
        run_validation();
        return;
    }

    let cases = test_cases();
    let inputs_dir = find_inputs_dir();
    let clang = find_clang();
    let clang_ver = clang.as_ref()
        .map(|c| clang_version(c))
        .unwrap_or_else(|| "not found".to_string());

    let has_clang = clang.is_some();

    if !has_clang {
        eprintln!("  WARNING: clang not found. Clang comparison columns will be empty.");
        eprintln!("  Install clang/Xcode command line tools for full comparison.");
    } else {
        eprintln!("  Using: {}", clang_ver);
    }

    let mut results = Vec::new();

    for case in &cases {
        eprint!("  Benchmarking {}...", case.name);

        // Build the LLVM2 function
        let func = (case.build_fn)();

        // LLVM2 benchmark
        let (llvm2_time, llvm2_inst_count, iters) = bench_llvm2(&func);

        // clang benchmark
        let c_file = inputs_dir.join(case.c_filename);
        let clang_time = clang.as_ref()
            .and_then(|c| time_clang(c, &c_file, CLANG_ITERS));
        let clang_inst_count = clang.as_ref()
            .and_then(|c| clang_instruction_count(c, &c_file));

        eprintln!(" done (LLVM2: {} iters, {}/func)",
            iters, format_duration_human(llvm2_time));

        results.push(CompareResult {
            name: case.name,
            description: case.description,
            llvm2_time,
            clang_time,
            llvm2_inst_count,
            clang_inst_count,
            llvm2_iters: iters,
        });
    }

    print_markdown_report(&results, &clang_ver, has_clang);
}

// ---------------------------------------------------------------------------
// Self-validation (run with --validate flag)
// ---------------------------------------------------------------------------

/// Validate that all test functions compile successfully through LLVM2.
fn run_validation() {
    let cases = test_cases();
    let mut all_ok = true;

    for case in &cases {
        eprint!("  Validating {}...", case.name);
        let func = (case.build_fn)();
        let (obj, count) = llvm2_compile(&func);
        if obj.is_empty() {
            eprintln!(" FAILED (empty object)");
            all_ok = false;
        } else if count == 0 {
            eprintln!(" FAILED (zero instructions)");
            all_ok = false;
        } else {
            eprintln!(" ok ({} instructions, {} bytes)", count, obj.len());
        }
    }

    // Validate ASM instruction counter
    {
        let asm = "\t.globl\t_trivial\n_trivial:\n\tmov\tx0, #42\n\tret\n";
        let count = count_asm_instructions(asm);
        eprint!("  Validating asm counter (trivial)...");
        if count != 2 {
            eprintln!(" FAILED (expected 2, got {})", count);
            all_ok = false;
        } else {
            eprintln!(" ok");
        }
    }
    {
        let asm = "\t.globl\t_max\n_max:\n\tcmp\tx0, x1\n\tcsel\tx0, x0, x1, gt\n\tret\n";
        let count = count_asm_instructions(asm);
        eprint!("  Validating asm counter (max)...");
        if count != 3 {
            eprintln!(" FAILED (expected 3, got {})", count);
            all_ok = false;
        } else {
            eprintln!(" ok");
        }
    }

    if !all_ok {
        eprintln!("\n  VALIDATION FAILED");
        std::process::exit(1);
    }
    eprintln!("\n  All validations passed.");
}
