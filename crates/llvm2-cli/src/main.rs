// llvm2-cli/main.rs - Command-line driver for LLVM2
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0
//
// Usage:
//   llvm2 input.tmbc -o output.o -O2 --target aarch64   (single binary file, object)
//   llvm2 a.tmbc b.tmbc c.tmbc -O2 -o prog              (multi-file, linked)
//   llvm2 -c a.tmbc b.tmbc                               (compile only, a.o b.o)
//   llvm2 -g -O2 -o prog a.tmbc                          (with debug info)
//   llvm2 --format=json module.json -o output.o          (debug-only JSON wire format)
//   llvm2 --format=text module.tmir -o output.o          (human-readable .tmir debug text)
//   llvm2 --format=auto module.tmbc                      (legacy auto-detect behaviour)
//   llvm2 --input-json module.json -o output.o           (DEPRECATED alias; use --format=json)
//   llvm2 --emit-tmir module.tmir input.tmbc             (dump parsed module as .tmir text)
//   llvm2 --version
//   llvm2 --help
//
// Input format rules (#414, tMIR transport architecture Layer 4):
//   - Binary `.tmbc` is the default and only accepted format by default.
//   - JSON is retained ONLY as a debug flag: pass `--format=json` to
//     enable it. The legacy `--input-json <FILE>` flag still works but
//     is deprecated and emits a warning.
//   - Pass `--format=auto` to restore the pre-#414 extension + magic
//     sniffing behaviour (useful for mixed-format test trees).
//
// Text format (.tmir, #413):
//   - `--format=text` reads a human-readable `.tmir` file via
//     `tmir::parser::parse_module`.
//   - `--emit-tmir <PATH>` writes the parsed module back out as
//     `.tmir` text (via `tmir::Module`'s Display impl), useful for
//     round-trip debugging. Like `--emit-json`, requires a single
//     input file.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{self, Command};

use clap::{Parser, ValueEnum};
use rayon::prelude::*;

use llvm2_codegen::compiler::{
    CompilationResult, Compiler, CompilerConfig, CompilerTraceLevel,
};

mod emit_proofs;
use llvm2_codegen::pipeline::{self, FormatMode, OptLevel};
use llvm2_codegen::target::Target;

/// Input format selector for the CLI `--format` flag.
///
/// See `designs/2026-04-16-tmir-transport-architecture.md` Layer 4 for
/// the binary/JSON story, and Layer 3 for the human-readable `.tmir`
/// text format added in #413.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum InputFormatArg {
    /// Binary tMIR bitcode (.tmbc). The default.
    Binary,
    /// JSON wire format (debug-only, opt-in).
    Json,
    /// Human-readable `.tmir` text format (debug-only, opt-in; #413).
    Text,
    /// Legacy auto-detect by extension + magic bytes.
    Auto,
}

impl InputFormatArg {
    fn to_mode(self) -> FormatMode {
        match self {
            InputFormatArg::Binary => FormatMode::Tmbc,
            InputFormatArg::Json => FormatMode::Json,
            InputFormatArg::Text => FormatMode::Text,
            InputFormatArg::Auto => FormatMode::Auto,
        }
    }
}

/// LLVM2 -- verified compiler backend for tMIR.
///
/// Compiles tMIR modules to Mach-O object files with optional
/// formal verification of each lowering step. Supports multi-file
/// compilation and system linker invocation.
#[derive(Parser, Debug)]
#[command(
    name = "llvm2",
    version = env!("CARGO_PKG_VERSION"),
    about = "LLVM2: verified compiler backend -- tMIR to machine code",
    long_about = "Verified compiler backend for the t* stack.\n\n\
        Compiles tMIR modules to native object files (Mach-O) with\n\
        proven-correct instruction lowering and optimizations.\n\n\
        Targets: aarch64 (primary), x86-64 (scaffold), riscv64 (scaffold).\n\n\
        Input format: binary tMIR bitcode (.tmbc) is the default.\n\
        JSON wire-format input is retained as a debug-only flag; pass\n\
        `--format=json` to enable it. Pass `--format=text` to read a\n\
        human-readable `.tmir` module (see issue #413). See issue #414\n\
        and the tMIR transport architecture design.\n\n\
        Examples:\n  \
        llvm2 -O2 -o prog a.tmbc b.tmbc      # compile + link (binary default)\n  \
        llvm2 -c a.tmbc b.tmbc                # compile only (.o)\n  \
        llvm2 -g -O2 -o prog a.tmbc           # with debug info\n  \
        llvm2 --format=json module.json       # debug JSON input (opt-in)\n  \
        llvm2 --format=text module.tmir       # human-readable .tmir debug text\n  \
        llvm2 --emit-tmir out.tmir in.tmbc    # dump module as .tmir text\n  \
        llvm2 --format=auto legacy.tmir       # legacy extension/magic sniffing"
)]
struct Cli {
    /// Input tMIR module files (positional). Default format is binary
    /// `.tmbc` bitcode; pass `--format=json` for the debug JSON wire
    /// format, or `--format=text` for human-readable `.tmir` text (#413).
    ///
    /// Multiple files are compiled in parallel and linked together.
    #[arg(value_name = "INPUT")]
    inputs: Vec<PathBuf>,

    /// Select the tMIR input format (`binary` default, `json` debug-only,
    /// `text` human-readable `.tmir` (#413), `auto` legacy sniffing).
    ///
    /// Per the tMIR transport architecture, binary `.tmbc` is the hot
    /// path for production tooling. JSON is retained solely for
    /// debugging and external-tool integration and must be enabled
    /// explicitly via `--format=json`. The human-readable `.tmir` text
    /// format is the canonical debug format for the t* stack and is
    /// enabled via `--format=text` (upstream `tmir::parser` feature).
    #[arg(long = "format", value_enum, default_value_t = InputFormatArg::Binary)]
    format: InputFormatArg,

    /// DEPRECATED: read a single tMIR module from a JSON wire format
    /// file. Prefer `--format=json <FILE>`. Retained for one release
    /// as an alias; emits a warning when used.
    ///
    /// Mutually exclusive with positional inputs.
    #[arg(long = "input-json", conflicts_with = "inputs")]
    input_json: Option<PathBuf>,

    /// Write the parsed tMIR module as JSON to this path (for round-trip testing).
    /// Only valid with a single input file or --input-json.
    #[arg(long = "emit-json")]
    emit_json: Option<PathBuf>,

    /// Write the parsed tMIR module as human-readable `.tmir` text to
    /// this path (for round-trip testing; #413).
    ///
    /// Uses `tmir::Module`'s `Display` impl (always on) to render the
    /// module. Only valid with a single input file. Complements
    /// `--emit-json` for quick diffing.
    #[arg(long = "emit-tmir")]
    emit_tmir: Option<PathBuf>,

    /// Output file path. For -c with a single input, this is the object file.
    /// Without -c, this is the linked executable.
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

    /// Compile only -- produce object files, do not link.
    #[arg(short = 'c')]
    compile_only: bool,

    /// Emit DWARF debug info sections in the output.
    #[arg(short = 'g')]
    debug_info: bool,

    /// Library search paths passed to the linker (-L <dir>).
    #[arg(short = 'L', value_name = "DIR")]
    lib_paths: Vec<PathBuf>,

    /// Libraries to link (-l <name>).
    #[arg(short = 'l', value_name = "LIB")]
    libs: Vec<String>,

    /// Emit proof certificates to the given directory (issue #421).
    ///
    /// When set, the compiler verifies every instruction with a matching
    /// proof obligation and writes one `.smt2` (SMT-LIB2 query) plus one
    /// `.cert` (JSON metadata) file per verified rule, organised under
    /// `<dir>/<ProofCategory>/<proof_name>.{smt2,cert}`.
    ///
    /// Example:
    ///   llvm2 -c --emit-proofs=proofs/ module.tmbc
    ///
    /// Consumers: `tla2` and `tRust` (issues #260, #269).
    /// Design: epic #407, task 6.
    #[arg(long = "emit-proofs", value_name = "DIR")]
    emit_proofs: Option<PathBuf>,

    /// Enable compilation trace output (per-phase timing).
    #[arg(long = "trace")]
    trace: bool,

    /// Print compilation metrics as JSON to stderr.
    #[arg(long = "metrics")]
    metrics: bool,

    /// Disable parallel per-function compilation within each module.
    ///
    /// By default, functions within a module are compiled in parallel using
    /// rayon. This flag disables that, compiling functions sequentially.
    /// Useful for debugging or when thread-safety issues are suspected.
    #[arg(long = "no-parallel")]
    no_parallel: bool,

    /// Enable CEGIS superoptimization with the given per-function budget
    /// in seconds (e.g. `--cegis-superopt=5`). Off by default.
    ///
    /// When set, the compiler runs the CEGIS-based superoptimization pass
    /// on each function with the given wall-clock budget. Results are keyed
    /// into a compilation cache so repeat compilations reuse proven
    /// rewrites. See issue #395 and
    /// `designs/2026-04-18-cache-and-cegis.md`.
    #[arg(long = "cegis-superopt", value_name = "SECS")]
    cegis_superopt: Option<u64>,

    /// Profile-generate mode: instrument the compiled module with
    /// basic-block counters and designate `<PATH>` as the destination
    /// for the resulting `.profdata` file. The runtime writes the file
    /// when the instrumented program exits.
    ///
    /// Phase 1 MVP — see `designs/2026-04-18-pgo-workflow.md`, issue #396.
    /// Counter injection is scaffolded (see [`llvm2_opt::pgo`]) but the
    /// full JIT-trampoline wiring + runtime dump will land in a later
    /// phase.
    #[arg(
        long = "profile-generate",
        value_name = "PATH",
        conflicts_with = "profile_use"
    )]
    profile_generate: Option<PathBuf>,

    /// Profile-use mode: load `<PATH>` as a `.profdata` file and stash
    /// the profile into the compilation pipeline. The profile is
    /// consumed by block layout, inline heuristic, loop unroll, and
    /// CEGIS budgeting (Phase 2).
    #[arg(long = "profile-use", value_name = "PATH")]
    profile_use: Option<PathBuf>,
}

fn parse_opt_level(s: &str) -> Result<OptLevel, String> {
    match s {
        "0" => Ok(OptLevel::O0),
        "1" => Ok(OptLevel::O1),
        "2" => Ok(OptLevel::O2),
        "3" => Ok(OptLevel::O3),
        _ => Err(format!(
            "invalid optimization level '{}': expected 0, 1, 2, or 3",
            s
        )),
    }
}

fn parse_target(s: &str) -> Result<Target, String> {
    match s.to_lowercase().as_str() {
        "aarch64" | "arm64" | "aarch64-apple-darwin" => Ok(Target::Aarch64),
        "x86_64" | "x86-64" | "x64" | "x86_64-apple-darwin" => Ok(Target::X86_64),
        "riscv64" | "riscv" => Ok(Target::Riscv64),
        _ => Err(format!(
            "unknown target '{}': supported targets are aarch64, x86-64, riscv64",
            s
        )),
    }
}

/// The result of compiling a single input file.
struct FileCompilationResult {
    /// Path to the generated .o file.
    object_path: PathBuf,
    /// Whether this is a temp file that should be cleaned up after linking.
    is_temp: bool,
    /// The compilation result from the compiler.
    result: CompilationResult,
}

/// Resolve the list of input file paths from CLI arguments.
fn resolve_inputs(cli: &Cli) -> Vec<PathBuf> {
    match (&cli.inputs.is_empty(), &cli.input_json) {
        (false, None) => cli.inputs.clone(),
        (true, Some(p)) => vec![p.clone()],
        (true, None) => {
            eprintln!("llvm2: error: no input files specified");
            eprintln!("  usage: llvm2 [OPTIONS] <INPUT>...                # binary .tmbc (default)");
            eprintln!("  usage: llvm2 [OPTIONS] --format=json <FILE>...   # JSON debug input");
            process::exit(1);
        }
        (false, Some(_)) => unreachable!("clap conflicts_with prevents this"),
    }
}

/// Resolve the effective input format.
///
/// `--input-json <FILE>` (deprecated) implies `--format=json` and
/// emits a one-line warning so existing callers learn the new flag.
/// Explicit `--format=<auto|binary|json>` always wins for positional
/// inputs; the two flags cannot be combined because clap marks
/// `--input-json` as `conflicts_with = inputs`, and the deprecated
/// path always means JSON.
fn resolve_format(cli: &Cli) -> FormatMode {
    if cli.input_json.is_some() {
        eprintln!(
            "llvm2: warning: `--input-json <FILE>` is deprecated; use \
             `--format=json <FILE>` instead (see issue #414)."
        );
        return FormatMode::Json;
    }
    cli.format.to_mode()
}

/// Compute the output .o path for a given input file in compile-only mode.
fn object_path_for(input: &PathBuf, output: Option<&PathBuf>, single: bool) -> PathBuf {
    if single {
        if let Some(out) = output {
            return out.clone();
        }
    }
    let mut p = input.clone();
    p.set_extension("o");
    p
}

/// Compute a temporary .o path for linking mode.
fn temp_object_path(input: &PathBuf, index: usize) -> PathBuf {
    let stem = input
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| format!("llvm2_input_{}", index));
    std::env::temp_dir().join(format!("llvm2_{}_{}.o", stem, std::process::id()))
}

/// Validate and echo the PGO flags.
///
/// Returns the parsed `.profdata` for `--profile-use`, if any. Terminates
/// the process with a clear error message if the file cannot be read or
/// fails header validation.
///
/// Phase 1: `--profile-generate` is recognised but the injection pass is
/// not yet wired into the codegen pipeline; we print a one-line note so
/// the user understands this is scaffolding. `--profile-use` fully loads
/// and validates the file so the CLI fails early on bad inputs.
fn handle_pgo_flags(cli: &Cli) -> Option<llvm2_opt::pgo::ProfData> {
    if let Some(ref out) = cli.profile_generate {
        eprintln!(
            "llvm2: --profile-generate recognised (target: {}); counter injection \
             is available via llvm2_opt::pgo (Phase 1 MVP, not yet wired into the \
             codegen pipeline). See designs/2026-04-18-pgo-workflow.md.",
            out.display(),
        );
    }
    if let Some(ref path) = cli.profile_use {
        match llvm2_opt::pgo::read_from_path(path) {
            Ok(p) => {
                eprintln!(
                    "llvm2: --profile-use: loaded {} function(s) from {}",
                    p.functions.len(),
                    path.display(),
                );
                return Some(p);
            }
            Err(e) => {
                eprintln!(
                    "llvm2: error: cannot load profile '{}': {}",
                    path.display(),
                    e,
                );
                process::exit(1);
            }
        }
    }
    None
}

/// Build a CompilerConfig from CLI flags.
fn build_config(cli: &Cli) -> CompilerConfig {
    CompilerConfig {
        opt_level: cli.opt_level,
        target: cli.target,
        emit_proofs: cli.emit_proofs.is_some(),
        trace_level: if cli.trace {
            CompilerTraceLevel::Full
        } else {
            CompilerTraceLevel::None
        },
        emit_debug: cli.debug_info,
        parallel: !cli.no_parallel,
        cegis_superopt_budget_sec: cli.cegis_superopt,
    }
}

/// Compile a single input file and return the result.
fn compile_one(
    input: &PathBuf,
    config: &CompilerConfig,
    format: FormatMode,
    obj_path: &PathBuf,
    is_temp: bool,
) -> FileCompilationResult {
    // Load the tMIR module using the explicit format selection (#414).
    let module: tmir::Module = match pipeline::load_module_as(input, format) {
        Ok(m) => m,
        Err(e) => {
            eprintln!(
                "llvm2: error: failed to read tMIR module from '{}': {}",
                input.display(),
                e,
            );
            process::exit(1);
        }
    };

    // Compile.
    let compiler = Compiler::new(config.clone());
    let result = match compiler.compile(&module) {
        Ok(r) => r,
        Err(e) => {
            eprintln!(
                "llvm2: error: compilation of '{}' failed: {}",
                input.display(),
                e,
            );
            process::exit(1);
        }
    };

    // Write the .o file.
    if let Err(e) = fs::write(obj_path, &result.object_code) {
        eprintln!(
            "llvm2: error: cannot write '{}': {}",
            obj_path.display(),
            e,
        );
        process::exit(1);
    }

    FileCompilationResult {
        object_path: obj_path.clone(),
        is_temp,
        result,
    }
}

/// Print trace information for a compilation result.
fn print_trace(result: &CompilationResult) {
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
}

/// Emit per-proof SMT-LIB2 + certificate files for a compilation result.
///
/// Implements the core of `--emit-proofs=<dir>` (issue #421). Errors are
/// reported on stderr but do not abort compilation — proof emission is a
/// side-channel audit feature, not a hard requirement for producing
/// object code.
fn emit_proof_artifacts(
    dir: &Path,
    result: &CompilationResult,
) -> Option<emit_proofs::EmitSummary> {
    let certs = match &result.proofs {
        Some(c) => c,
        None => {
            eprintln!(
                "llvm2: warning: --emit-proofs set but compiler produced no certificates"
            );
            return None;
        }
    };

    match emit_proofs::emit_proof_files(dir, certs.as_slice()) {
        Ok(summary) => Some(summary),
        Err(e) => {
            eprintln!(
                "llvm2: error: failed to emit proof files to '{}': {}",
                dir.display(),
                e,
            );
            None
        }
    }
}

/// Print proof certificates for a compilation result.
fn print_proofs(result: &CompilationResult) {
    if let Some(ref proofs) = result.proofs {
        if proofs.is_empty() {
            eprintln!("llvm2: note: proof emission enabled but no certificates produced (z4 not yet integrated)");
        } else {
            eprintln!("--- proof certificates ---");
            for cert in proofs {
                let status = if cert.verified {
                    "VERIFIED"
                } else {
                    "UNVERIFIED"
                };
                eprintln!("  {} [{}]", cert.rule_name, status);
            }
            eprintln!("--- end proofs ---");
        }
    }
}

/// Print compilation metrics for a compilation result.
fn print_metrics(result: &CompilationResult) {
    let metrics_json = serde_json::json!({
        "code_size_bytes": result.metrics.code_size_bytes,
        "instruction_count": result.metrics.instruction_count,
        "function_count": result.metrics.function_count,
        "optimization_passes_run": result.metrics.optimization_passes_run,
    });
    eprintln!("{}", serde_json::to_string_pretty(&metrics_json).unwrap());
}

/// Map target architecture to linker -arch flag value.
fn linker_arch(target: Target) -> &'static str {
    match target {
        Target::Aarch64 => "arm64",
        Target::X86_64 => "x86_64",
        Target::Riscv64 => "riscv64",
    }
}

/// Invoke the system linker to combine object files into an executable.
fn link(
    object_files: &[PathBuf],
    output: &PathBuf,
    target: Target,
    lib_paths: &[PathBuf],
    libs: &[String],
) {
    let mut cmd = Command::new("cc");

    cmd.arg("-o").arg(output);
    cmd.arg("-arch").arg(linker_arch(target));

    for obj in object_files {
        cmd.arg(obj);
    }

    for dir in lib_paths {
        cmd.arg(format!("-L{}", dir.display()));
    }

    for lib in libs {
        cmd.arg(format!("-l{}", lib));
    }

    eprintln!(
        "llvm2: linking {} object file(s) -> {}",
        object_files.len(),
        output.display(),
    );

    match cmd.status() {
        Ok(status) if status.success() => {}
        Ok(status) => {
            eprintln!(
                "llvm2: error: linker exited with status {}",
                status.code().unwrap_or(-1),
            );
            process::exit(1);
        }
        Err(e) => {
            eprintln!("llvm2: error: failed to invoke linker 'cc': {}", e);
            eprintln!(
                "  hint: ensure Xcode command line tools are installed (xcode-select --install)"
            );
            process::exit(1);
        }
    }
}

/// Clean up temporary object files.
fn cleanup_temps(files: &[FileCompilationResult]) {
    for f in files {
        if f.is_temp {
            let _ = fs::remove_file(&f.object_path);
        }
    }
}

fn main() {
    let cli = Cli::parse();
    let inputs = resolve_inputs(&cli);
    let format = resolve_format(&cli);

    // Validate: --emit-json / --emit-tmir only make sense with a single input.
    if cli.emit_json.is_some() && inputs.len() > 1 {
        eprintln!("llvm2: error: --emit-json requires exactly one input file");
        process::exit(1);
    }
    if cli.emit_tmir.is_some() && inputs.len() > 1 {
        eprintln!("llvm2: error: --emit-tmir requires exactly one input file");
        process::exit(1);
    }

    // Emit JSON round-trip output if requested (single input only).
    if let Some(ref emit_path) = cli.emit_json {
        let module: tmir::Module = match pipeline::load_module_as(&inputs[0], format) {
            Ok(m) => m,
            Err(e) => {
                eprintln!(
                    "llvm2: error: failed to read tMIR module from '{}': {}",
                    inputs[0].display(),
                    e,
                );
                process::exit(1);
            }
        };
        match serde_json::to_string_pretty(&module)
            .map_err(|e| e.to_string())
            .and_then(|json| fs::write(emit_path, json).map_err(|e| e.to_string()))
        {
            Ok(()) => {
                eprintln!("llvm2: wrote tMIR JSON to {}", emit_path.display());
            }
            Err(e) => {
                eprintln!(
                    "llvm2: error: failed to write tMIR JSON to '{}': {}",
                    emit_path.display(),
                    e,
                );
                process::exit(1);
            }
        }
    }

    // Emit .tmir text round-trip output if requested (#413, single input only).
    if let Some(ref emit_path) = cli.emit_tmir {
        let module: tmir::Module = match pipeline::load_module_as(&inputs[0], format) {
            Ok(m) => m,
            Err(e) => {
                eprintln!(
                    "llvm2: error: failed to read tMIR module from '{}': {}",
                    inputs[0].display(),
                    e,
                );
                process::exit(1);
            }
        };
        match pipeline::save_module_to_tmir_text(&module, emit_path) {
            Ok(()) => {
                eprintln!("llvm2: wrote tMIR text to {}", emit_path.display());
            }
            Err(e) => {
                eprintln!(
                    "llvm2: error: failed to write tMIR text to '{}': {}",
                    emit_path.display(),
                    e,
                );
                process::exit(1);
            }
        }
    }

    // Parse PGO flags. `--profile-use` is fully validated here so we fail
    // early on bad profiles; the loaded `ProfData` will be threaded into
    // the pipeline via `OptContext::profile` once that lands (Phase 2).
    let _loaded_profile = handle_pgo_flags(&cli);

    let config = build_config(&cli);

    // Determine whether we need temp files (linking mode) or permanent files
    // (compile-only mode).
    let compile_only = cli.compile_only;

    let file_results: Vec<FileCompilationResult> = if inputs.len() == 1 {
        // Single file: no need for rayon overhead.
        let input = &inputs[0];
        let obj_path = if compile_only {
            object_path_for(input, cli.output.as_ref(), true)
        } else {
            temp_object_path(input, 0)
        };
        let is_temp = !compile_only;
        vec![compile_one(input, &config, format, &obj_path, is_temp)]
    } else {
        // Multiple files: parallel compilation via rayon.
        inputs
            .par_iter()
            .enumerate()
            .map(|(i, input)| {
                let obj_path = if compile_only {
                    object_path_for(input, None, false)
                } else {
                    temp_object_path(input, i)
                };
                let is_temp = !compile_only;
                compile_one(input, &config, format, &obj_path, is_temp)
            })
            .collect()
    };

    // Print per-file diagnostics.
    let mut total_functions = 0usize;
    let mut total_code_bytes = 0usize;
    let mut total_emit_summary = emit_proofs::EmitSummary::default();

    for fr in &file_results {
        print_trace(&fr.result);
        print_proofs(&fr.result);
        if cli.metrics {
            print_metrics(&fr.result);
        }

        // Emit per-proof SMT-LIB2 + certificate files (#421).
        if let Some(ref dir) = cli.emit_proofs {
            if let Some(s) = emit_proof_artifacts(dir, &fr.result) {
                total_emit_summary.merge(s);
            }
        }

        total_functions += fr.result.metrics.function_count;
        total_code_bytes += fr.result.metrics.code_size_bytes;

        if compile_only {
            eprintln!(
                "llvm2: compiled {} function(s), {} bytes -> {}",
                fr.result.metrics.function_count,
                fr.result.metrics.code_size_bytes,
                fr.object_path.display(),
            );
        }
    }

    if cli.emit_proofs.is_some() {
        eprintln!(
            "llvm2: wrote {} .smt2 + {} .cert file(s) to {} ({} certs had no obligation in the database)",
            total_emit_summary.smt2_written,
            total_emit_summary.cert_written,
            cli.emit_proofs.as_ref().unwrap().display(),
            total_emit_summary.skipped_no_obligation,
        );
    }

    if compile_only {
        // Done -- object files are in place.
        return;
    }

    // Linking mode: combine all .o files into an executable.
    let output_path = cli.output.unwrap_or_else(|| PathBuf::from("a.out"));

    let object_paths: Vec<PathBuf> = file_results.iter().map(|f| f.object_path.clone()).collect();

    link(
        &object_paths,
        &output_path,
        cli.target,
        &cli.lib_paths,
        &cli.libs,
    );

    // Clean up temp .o files.
    cleanup_temps(&file_results);

    eprintln!(
        "llvm2: linked {} file(s), {} function(s), {} code bytes -> {}",
        file_results.len(),
        total_functions,
        total_code_bytes,
        output_path.display(),
    );
}
