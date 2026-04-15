// llvm2-cli/main.rs - Command-line driver for LLVM2
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Usage:
//   llvm2 input.json -o output.o -O2 --target aarch64
//   llvm2 --input-json module.json -o output.o    (validated JSON wire format)
//   llvm2 --input-json module.json --emit-json roundtrip.json  (round-trip)
//   llvm2 --version
//   llvm2 --help

use std::fs;
use std::path::PathBuf;
use std::process;

use clap::Parser;

use llvm2_codegen::compiler::{Compiler, CompilerConfig, CompilerTraceLevel};
use llvm2_codegen::pipeline::OptLevel;
use llvm2_codegen::target::Target;
use tmir_func::reader;

/// LLVM2 -- verified compiler backend for tMIR.
///
/// Compiles tMIR modules to Mach-O object files with optional
/// formal verification of each lowering step.
#[derive(Parser, Debug)]
#[command(
    name = "llvm2",
    version = env!("CARGO_PKG_VERSION"),
    about = "LLVM2: verified compiler backend -- tMIR to machine code",
    long_about = "Verified compiler backend for the t* stack.\n\n\
        Compiles tMIR modules to native object files (Mach-O) with\n\
        proven-correct instruction lowering and optimizations.\n\n\
        Targets: aarch64 (primary), x86-64 (scaffold), riscv64 (scaffold)."
)]
struct Cli {
    /// Input tMIR module file (positional, JSON-serialized tmir_func::Module).
    /// Use --input-json for explicit JSON wire format reading with validation.
    input: Option<PathBuf>,

    /// Read tMIR module from a JSON wire format file (with structural validation).
    ///
    /// This is the recommended way to pass tMIR from external tools.
    /// The JSON format matches serde serialization of tmir_func::Module.
    #[arg(long = "input-json", conflicts_with = "input")]
    input_json: Option<PathBuf>,

    /// Write the parsed tMIR module as JSON to this path (for round-trip testing).
    #[arg(long = "emit-json")]
    emit_json: Option<PathBuf>,

    /// Output object file path.
    #[arg(short = 'o', long = "output")]
    output: Option<PathBuf>,

    /// Optimization level.
    #[arg(
        short = 'O',
        long = "opt-level",
        value_parser = parse_opt_level,
        default_value = "2"
    )]
    opt_level: OptLevel,

    /// Target architecture.
    #[arg(long = "target", value_parser = parse_target, default_value = "aarch64")]
    target: Target,

    /// Emit proof certificates for each lowering rule (placeholder for z4).
    #[arg(long = "emit-proofs")]
    emit_proofs: bool,

    /// Enable compilation trace output (per-phase timing).
    #[arg(long = "trace")]
    trace: bool,

    /// Print compilation metrics as JSON to stderr.
    #[arg(long = "metrics")]
    metrics: bool,
}

fn parse_opt_level(s: &str) -> Result<OptLevel, String> {
    match s {
        "0" => Ok(OptLevel::O0),
        "1" => Ok(OptLevel::O1),
        "2" => Ok(OptLevel::O2),
        "3" => Ok(OptLevel::O3),
        _ => Err(format!("invalid optimization level '{}': expected 0, 1, 2, or 3", s)),
    }
}

fn parse_target(s: &str) -> Result<Target, String> {
    match s.to_lowercase().as_str() {
        "aarch64" | "arm64" => Ok(Target::Aarch64),
        "x86_64" | "x86-64" | "x64" => Ok(Target::X86_64),
        "riscv64" | "riscv" => Ok(Target::Riscv64),
        _ => Err(format!(
            "unknown target '{}': supported targets are aarch64, x86-64, riscv64",
            s
        )),
    }
}

fn main() {
    let cli = Cli::parse();

    // Determine the input file path.
    let input_path = match (&cli.input, &cli.input_json) {
        (Some(p), None) => p.clone(),
        (None, Some(p)) => p.clone(),
        (None, None) => {
            eprintln!("llvm2: error: no input file specified (use positional argument or --input-json)");
            process::exit(1);
        }
        (Some(_), Some(_)) => unreachable!("clap conflicts_with prevents this"),
    };

    // Read and deserialize the tMIR module.
    // --input-json uses the validated reader; positional uses raw serde.
    let module: tmir_func::Module = if cli.input_json.is_some() {
        match reader::read_module_from_json(&input_path) {
            Ok(m) => m,
            Err(e) => {
                eprintln!(
                    "llvm2: error: failed to read tMIR JSON from '{}': {}",
                    input_path.display(),
                    e
                );
                process::exit(1);
            }
        }
    } else {
        let input_bytes = match fs::read(&input_path) {
            Ok(bytes) => bytes,
            Err(e) => {
                eprintln!("llvm2: error: cannot read '{}': {}", input_path.display(), e);
                process::exit(1);
            }
        };
        match serde_json::from_slice(&input_bytes) {
            Ok(m) => m,
            Err(e) => {
                eprintln!(
                    "llvm2: error: failed to parse '{}' as tMIR module: {}",
                    input_path.display(),
                    e
                );
                process::exit(1);
            }
        }
    };

    // Emit JSON round-trip output if requested.
    if let Some(ref emit_path) = cli.emit_json {
        match reader::write_module_to_json(&module, emit_path) {
            Ok(()) => {
                eprintln!("llvm2: wrote tMIR JSON to {}", emit_path.display());
            }
            Err(e) => {
                eprintln!(
                    "llvm2: error: failed to write tMIR JSON to '{}': {}",
                    emit_path.display(),
                    e
                );
                process::exit(1);
            }
        }
    }

    // Build compiler configuration from CLI flags.
    let config = CompilerConfig {
        opt_level: cli.opt_level,
        target: cli.target,
        emit_proofs: cli.emit_proofs,
        trace_level: if cli.trace {
            CompilerTraceLevel::Full
        } else {
            CompilerTraceLevel::None
        },
    };

    let compiler = Compiler::new(config);

    // Compile the tMIR module.
    let result = match compiler.compile(&module) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("llvm2: error: compilation failed: {}", e);
            process::exit(1);
        }
    };

    // Print trace if requested.
    if let Some(ref trace) = result.trace {
        eprintln!("--- compilation trace ---");
        for entry in &trace.entries {
            let detail = entry.detail.as_deref().unwrap_or("");
            eprintln!(
                "  {:20} {:>8.2}ms  {}",
                entry.phase,
                entry.duration.as_secs_f64() * 1000.0,
                detail,
            );
        }
        eprintln!(
            "  {:20} {:>8.2}ms",
            "TOTAL",
            trace.total_duration.as_secs_f64() * 1000.0,
        );
        eprintln!("--- end trace ---");
    }

    // Print proof certificates if requested.
    if let Some(ref proofs) = result.proofs {
        if proofs.is_empty() {
            eprintln!("llvm2: note: proof emission enabled but no certificates produced (z4 not yet integrated)");
        } else {
            eprintln!("--- proof certificates ---");
            for cert in proofs {
                let status = if cert.verified { "VERIFIED" } else { "UNVERIFIED" };
                eprintln!("  {} [{}]", cert.rule_name, status);
            }
            eprintln!("--- end proofs ---");
        }
    }

    // Print metrics if requested.
    if cli.metrics {
        let metrics_json = serde_json::json!({
            "code_size_bytes": result.metrics.code_size_bytes,
            "instruction_count": result.metrics.instruction_count,
            "function_count": result.metrics.function_count,
            "optimization_passes_run": result.metrics.optimization_passes_run,
        });
        eprintln!("{}", serde_json::to_string_pretty(&metrics_json).unwrap());
    }

    // Determine output path: explicit -o, or derive from input.
    let output_path = cli.output.unwrap_or_else(|| {
        let mut p = input_path.clone();
        p.set_extension("o");
        p
    });

    // Write the Mach-O object file.
    if let Err(e) = fs::write(&output_path, &result.object_code) {
        eprintln!(
            "llvm2: error: cannot write '{}': {}",
            output_path.display(),
            e
        );
        process::exit(1);
    }

    eprintln!(
        "llvm2: compiled {} function(s), {} bytes -> {}",
        result.metrics.function_count,
        result.metrics.code_size_bytes,
        output_path.display(),
    );
}
