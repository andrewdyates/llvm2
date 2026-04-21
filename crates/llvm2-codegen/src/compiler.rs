// llvm2-codegen/compiler.rs - Public compilation API
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0

//! Public compilation library API for LLVM2.
//!
//! The [`Compiler`] struct is the top-level entry point for programmatic
//! compilation. It wraps the internal [`Pipeline`](crate::pipeline::Pipeline)
//! with a clean configuration interface and structured result types.
//!
//! # Quick start
//!
//! ```ignore
//! use llvm2_codegen::compiler::{Compiler, CompilerConfig};
//!
//! let compiler = Compiler::new(CompilerConfig::default());
//! let result = compiler.compile(&tmir_module)?;
//! let object_bytes = result.object_code;
//! ```

use std::time::{Duration, Instant};

use rayon::prelude::*;
use thiserror::Error;

use crate::dialect_pipeline::{DialectPipelineError, lower_dialects};
use crate::pipeline::{OptLevel, Pipeline, PipelineConfig, PipelineError};
use crate::target::Target;

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// Errors that can occur during compilation through the [`Compiler`] API.
#[derive(Debug, Error)]
pub enum CompileError {
    /// tMIR to LIR adapter translation failed.
    #[error("adapter error: {0}")]
    Adapter(#[from] llvm2_lower::AdapterError),

    /// Pre-adapter tMIR dialect lowering failed (unknown dialect op, pass
    /// error, or fixpoint not reached). See `dialect_pipeline` module.
    /// Tracked under #433 / tMIR #428.
    #[error("dialect pipeline error: {0}")]
    DialectPipeline(#[from] DialectPipelineError),

    /// Pipeline compilation failed (ISel, regalloc, encoding, etc.).
    #[error("pipeline error: {0}")]
    Pipeline(#[from] PipelineError),

    /// Module contains no functions.
    #[error("empty module: no functions to compile")]
    EmptyModule,

    /// JIT compilation failed while producing executable memory.
    #[error("JIT compilation failed: {0}")]
    Jit(#[from] crate::jit::JitError),

    /// Caller requested proof certificates (`config.emit_proofs == true`) for
    /// a target that does not yet produce per-instruction proof certificates
    /// through a target-specific function verifier.
    ///
    /// Returned instead of silently producing `proofs: None` so embedders that
    /// opt into a verified-codegen attestation workflow learn the truth at the
    /// public API boundary rather than trusting a silent lie. Today this fires
    /// for `Target::Riscv64` unconditionally, and for `Target::X86_64` only
    /// when the `verify` feature is disabled (with `verify` on, the x86-64
    /// path produces real certs via `llvm2_verify::x86_64_function_verifier`
    /// — see #465). The AArch64 path always emits certificates.
    ///
    /// Tracking: #465 (x86-64 proofs; landed for default `verify` builds),
    /// RISC-V proofs (TBD).
    #[error(
        "proof certificates are not yet supported for target {target:?} in \
         this build; set config.emit_proofs = false, or compile for \
         Target::Aarch64, or enable the `verify` feature for Target::X86_64. \
         Tracking: #465 (x86-64 proofs), RISC-V proofs (TBD)."
    )]
    ProofsUnsupportedForTarget {
        /// The target that requested proofs but cannot produce them.
        target: Target,
    },
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Verbosity level for the compiler's timing trace.
///
/// Distinct from [`llvm2_ir::TraceLevel`] which controls structured event
/// logging. This enum controls per-phase timing output in the compiler API.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompilerTraceLevel {
    /// No trace output.
    None,
    /// Summary: total time and pass counts.
    Summary,
    /// Full: per-phase timing and per-function details.
    Full,
}

/// Configuration for the [`Compiler`].
#[derive(Debug, Clone)]
pub struct CompilerConfig {
    /// Optimization level (O0 through O3).
    pub opt_level: OptLevel,
    /// Target architecture.
    pub target: Target,
    /// Whether to emit proof certificates for each compiled function.
    ///
    /// When true, the compiler runs the llvm2-verify function verifier on
    /// each prepared MachFunction and produces a [`ProofCertificate`] per
    /// verified instruction. Currently uses mock evaluation (exhaustive for
    /// 8-bit, statistical for 32/64-bit); will upgrade to formal z4 proofs
    /// when z4 integration is complete.
    pub emit_proofs: bool,
    /// Compilation trace verbosity.
    pub trace_level: CompilerTraceLevel,
    /// Whether to emit DWARF debug info sections in the output object file.
    ///
    /// When true, the pipeline generates `__debug_info`, `__debug_abbrev`,
    /// `__debug_str`, and `__debug_line` sections.
    pub emit_debug: bool,
    /// Whether to compile functions in parallel using rayon.
    ///
    /// When true and the module contains 2+ functions, the per-function
    /// compilation phases (ISel, optimization, register allocation, frame
    /// lowering, branch resolution) run in parallel across a rayon thread
    /// pool. The final Mach-O emission phase remains sequential since it
    /// builds a single combined `__text` section.
    ///
    /// Default: `true`.
    pub parallel: bool,
    /// Per-function wall-clock budget for the CEGIS superopt pass.
    ///
    /// When `Some(n)`, the pipeline runs [`llvm2_verify::CegisSuperoptPass`]
    /// with `budget_sec = n` during optimization. Results are keyed into the
    /// compilation cache so repeat compilations reuse proven rewrites.
    /// When `None` (default), the pass is not scheduled. The CLI flag
    /// `--cegis-superopt=<secs>` sets this field.
    ///
    /// Issue: #395. Default: `None` (off).
    pub cegis_superopt_budget_sec: Option<u64>,
}

impl Default for CompilerConfig {
    fn default() -> Self {
        Self {
            opt_level: OptLevel::O2,
            target: Target::Aarch64,
            emit_proofs: false,
            trace_level: CompilerTraceLevel::None,
            emit_debug: false,
            parallel: true,
            cegis_superopt_budget_sec: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// Metrics collected during compilation.
#[derive(Debug, Clone, Default)]
pub struct CompilationMetrics {
    /// Total size of emitted machine code in bytes.
    pub code_size_bytes: usize,
    /// Total number of machine instructions emitted (across all functions).
    pub instruction_count: usize,
    /// Number of functions compiled.
    pub function_count: usize,
    /// Number of optimization passes executed.
    pub optimization_passes_run: usize,
}

/// Per-function code-quality metrics. Surfaced through
/// [`JitCompilationResult::per_function_metrics`] so callers can identify
/// functions with high spill pressure, unusual branch density, etc.
///
/// See issue #364 item 3.
#[derive(Debug, Clone, Default)]
pub struct FunctionQualityMetrics {
    pub name: String,
    /// Real (non-pseudo) machine instructions emitted.
    pub instruction_count: usize,
    /// Number of stack slots allocated by register allocation for spills.
    /// Not the sum of spill stores emitted — this counts distinct spill
    /// slots; each slot can be reused when spills don't overlap
    /// (`enable_spill_slot_reuse`).
    pub spill_slot_count: usize,
    /// Number of branch-terminator instructions (B, B.cond, CBZ, TBZ, BL
    /// for tail calls, RET). Used as a coarse branch-density proxy.
    pub branch_count: usize,
    /// Phase timings captured during `prepare_function_with_metrics`.
    pub phase_timings: crate::pipeline::PhaseTimings,
}

/// A single phase entry in a compilation trace.
#[derive(Debug, Clone)]
pub struct TraceEntry {
    /// Name of the compilation phase (e.g., "adapter", "isel", "regalloc").
    pub phase: String,
    /// Wall-clock duration of this phase.
    pub duration: Duration,
    /// Optional detail (e.g., function name being compiled).
    pub detail: Option<String>,
}

/// Compiler-level trace containing per-phase timing information.
///
/// Distinct from [`llvm2_ir::CompilationTrace`] which is a thread-safe
/// structured event collector for instruction-level provenance.
/// This struct is a simple timing log for the compiler API.
#[derive(Debug, Clone, Default)]
pub struct CompilerTrace {
    /// Ordered list of phase entries.
    pub entries: Vec<TraceEntry>,
    /// Total wall-clock compilation time.
    pub total_duration: Duration,
}

/// A proof certificate for a single lowering rule.
///
/// Each certificate records the result of verifying one instruction's
/// lowering against its proof obligation from the [`ProofDatabase`].
/// When `verified` is true, the proof obligation passed (exhaustive for
/// small bit-widths, statistical sampling for 32/64-bit). When z4
/// integration is complete, formal SMT proofs will upgrade certificates
/// to [`llvm2_verify::VerificationStrength::Formal`].
#[derive(Debug, Clone)]
pub struct ProofCertificate {
    /// Name of the verified lowering rule (e.g., "iadd_i32").
    pub rule_name: String,
    /// Whether the proof was successfully verified.
    pub verified: bool,
    /// Proof category (e.g., Arithmetic, Memory, Branch).
    pub category: String,
    /// Verification strength achieved (e.g., "Exhaustive", "Statistical(100000)").
    pub strength: String,
    /// Name of the function this instruction belongs to.
    pub function_name: String,
}

/// The result of compiling a tMIR module through the LLVM2 pipeline.
#[derive(Debug, Clone)]
pub struct CompilationResult {
    /// Mach-O object file bytes containing all compiled functions.
    pub object_code: Vec<u8>,
    /// Compilation metrics.
    pub metrics: CompilationMetrics,
    /// Optional compiler trace (populated when trace_level != None).
    pub trace: Option<CompilerTrace>,
    /// Optional proof certificates (populated when emit_proofs is true).
    pub proofs: Option<Vec<ProofCertificate>>,
}

/// The result of compiling a tMIR module through the LLVM2 pipeline to
/// executable memory for JIT execution.
///
/// Unlike [`CompilationResult`] which produces Mach-O object file bytes,
/// this result contains an [`ExecutableBuffer`](crate::jit::ExecutableBuffer)
/// with all functions linked and ready for immediate execution.
pub struct JitCompilationResult {
    /// Executable memory buffer containing all compiled functions.
    pub buffer: crate::jit::ExecutableBuffer,
    /// Compilation metrics.
    pub metrics: CompilationMetrics,
    /// Optional compiler trace (populated when trace_level != None).
    pub trace: Option<CompilerTrace>,
    /// Optional proof certificates (populated when emit_proofs is true).
    pub proofs: Option<Vec<ProofCertificate>>,
    /// Per-function code-quality and phase-timing metrics. See #364.
    /// Always populated (one entry per compiled function).
    pub per_function_metrics: Vec<FunctionQualityMetrics>,
}

impl std::fmt::Debug for JitCompilationResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JitCompilationResult")
            .field("buffer_size", &self.buffer.allocated_size())
            .field("symbol_count", &self.buffer.symbol_count())
            .field("metrics", &self.metrics)
            .field("trace", &self.trace)
            .field("proofs", &self.proofs)
            .field(
                "per_function_metrics_count",
                &self.per_function_metrics.len(),
            )
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Count the number of real (non-pseudo) machine instructions in a function.
///
/// Walks blocks in layout order and counts instructions that will actually be
/// encoded to machine code. Pseudo-instructions (Phi, Copy, StackAlloc, Nop,
/// etc.) are excluded since they have no hardware encoding.
fn count_real_instructions(func: &llvm2_ir::MachFunction) -> usize {
    func.block_order
        .iter()
        .map(|&block_id| {
            func.blocks[block_id.0 as usize]
                .insts
                .iter()
                .filter(|&&inst_id| !func.insts[inst_id.0 as usize].is_pseudo())
                .count()
        })
        .sum()
}

/// Count branch-like instructions for coarse per-function density metrics.
///
/// The JIT path currently prepares AArch64 `MachFunction`s, so this helper
/// recognizes AArch64 branch / return opcodes directly and also counts any
/// non-pseudo terminator flagged in the IR.
fn count_branch_instructions(func: &llvm2_ir::MachFunction) -> usize {
    func.block_order
        .iter()
        .map(|&block_id| {
            func.blocks[block_id.0 as usize]
                .insts
                .iter()
                .filter(|&&inst_id| {
                    let inst = &func.insts[inst_id.0 as usize];
                    !inst.is_pseudo()
                        && (inst.is_terminator()
                            || matches!(
                                inst.opcode,
                                llvm2_ir::inst::AArch64Opcode::B
                                    | llvm2_ir::inst::AArch64Opcode::BCond
                                    | llvm2_ir::inst::AArch64Opcode::Bcc
                                    | llvm2_ir::inst::AArch64Opcode::Cbz
                                    | llvm2_ir::inst::AArch64Opcode::Cbnz
                                    | llvm2_ir::inst::AArch64Opcode::Tbz
                                    | llvm2_ir::inst::AArch64Opcode::Tbnz
                                    | llvm2_ir::inst::AArch64Opcode::Br
                                    | llvm2_ir::inst::AArch64Opcode::Bl
                                    | llvm2_ir::inst::AArch64Opcode::BL
                                    | llvm2_ir::inst::AArch64Opcode::Ret
                            ))
                })
                .count()
        })
        .sum()
}

// ---------------------------------------------------------------------------
// Compiler
// ---------------------------------------------------------------------------

/// The LLVM2 compiler — top-level API for compiling tMIR to machine code.
///
/// Wraps the internal [`Pipeline`] with a clean configuration interface,
/// structured results, optional tracing, and proof certificate emission.
pub struct Compiler {
    config: CompilerConfig,
}

/// Generate proof certificates for a MachFunction by running the function
/// verifier from llvm2-verify. Each verified instruction produces a
/// certificate recording the proof obligation name, category, and strength.
///
/// Only available when the `verify` feature is enabled.
#[cfg(feature = "verify")]
fn generate_proof_certificates(func: &llvm2_ir::MachFunction) -> Vec<ProofCertificate> {
    use llvm2_verify::function_verifier::{InstructionVerificationResult, verify_function};

    let report = verify_function(func);
    let mut certs = Vec::new();

    for inst_report in &report.instructions {
        match &inst_report.result {
            InstructionVerificationResult::Verified {
                proof_name,
                category,
                strength,
            } => {
                certs.push(ProofCertificate {
                    rule_name: proof_name.clone(),
                    verified: true,
                    category: format!("{}", category),
                    strength: format!("{:?}", strength),
                    function_name: report.function_name.clone(),
                });
            }
            InstructionVerificationResult::Failed { proof_name, detail } => {
                certs.push(ProofCertificate {
                    rule_name: proof_name.clone(),
                    verified: false,
                    category: String::new(),
                    strength: format!("Failed: {}", detail),
                    function_name: report.function_name.clone(),
                });
            }
            // Skipped (pseudo-ops) and Unverified (no proof mapping) do not
            // produce certificates — they are not claims of correctness.
            _ => {}
        }
    }

    certs
}

/// x86-64 mirror of [`generate_proof_certificates`] (#465).
///
/// Walks an [`llvm2_lower::x86_64_isel::X86ISelFunction`] via the
/// [`llvm2_verify::X86FunctionVerifier`] and emits a [`ProofCertificate`]
/// per verified instruction. Output shape matches the AArch64 path so the
/// public `CompilationResult::proofs` vector is target-agnostic.
#[cfg(feature = "verify")]
fn generate_x86_64_proof_certificates(
    func: &llvm2_lower::x86_64_isel::X86ISelFunction,
) -> Vec<ProofCertificate> {
    use llvm2_verify::function_verifier::InstructionVerificationResult;
    use llvm2_verify::verify_x86_64_function;

    let report = verify_x86_64_function(func);
    let mut certs = Vec::new();

    for inst_report in &report.instructions {
        match &inst_report.result {
            InstructionVerificationResult::Verified {
                proof_name,
                category,
                strength,
            } => {
                certs.push(ProofCertificate {
                    rule_name: proof_name.clone(),
                    verified: true,
                    category: format!("{}", category),
                    strength: format!("{:?}", strength),
                    function_name: report.function_name.clone(),
                });
            }
            InstructionVerificationResult::Failed { proof_name, detail } => {
                certs.push(ProofCertificate {
                    rule_name: proof_name.clone(),
                    verified: false,
                    category: String::new(),
                    strength: format!("Failed: {}", detail),
                    function_name: report.function_name.clone(),
                });
            }
            // Skipped (pseudo-ops) and Unverified (no proof mapping) do not
            // produce certificates — same policy as the AArch64 path.
            _ => {}
        }
    }

    certs
}

impl Compiler {
    /// Create a new compiler with the given configuration.
    pub fn new(config: CompilerConfig) -> Self {
        Self { config }
    }

    /// Create a compiler with default O2 configuration targeting AArch64.
    pub fn default_o2() -> Self {
        Self::new(CompilerConfig::default())
    }

    /// Returns the compiler's current configuration.
    pub fn config(&self) -> &CompilerConfig {
        &self.config
    }

    /// Compile a tMIR module to a Mach-O object file.
    ///
    /// Translates each function in the module through the full pipeline:
    /// tMIR adapter -> ISel -> optimization -> regalloc -> frame lowering
    /// -> encoding -> Mach-O emission.
    ///
    /// All functions are compiled into a single Mach-O object file with each
    /// function as a separate symbol in the `__text` section. Cross-function
    /// calls are represented as relocations for the linker.
    ///
    /// Returns the compiled object code, metrics, optional trace, and
    /// optional proof certificates.
    pub fn compile(&self, module: &tmir::Module) -> Result<CompilationResult, CompileError> {
        let total_start = Instant::now();
        let tracing = self.config.trace_level != CompilerTraceLevel::None;
        let mut trace_entries = Vec::new();

        // Phase 0: Pre-adapter dialect lowering (#433, tMIR #428).
        //
        // Runs `tmir::dialect::lower_module` with an internal
        // `DialectRegistry` so any `Inst::DialectOp` (e.g. `verif.bfs_step`,
        // `verif.frontier_drain`) is rewritten into core tMIR before the
        // adapter runs. Unknown dialects are rejected here — the adapter
        // has no DialectOp handler and would otherwise fail at ISel.
        //
        // Clones the module locally because the dialect driver needs
        // `&mut Module` and the public `compile` signature is `&Module`.
        // The clone is O(module) and only traverses the IR once; if this
        // ever shows up in a profile we can lift the path to a
        // `compile_owned` API instead of cloning.
        let dialect_start = Instant::now();
        let mut lowered_module = module.clone();
        let rewrites = lower_dialects(&mut lowered_module)?;
        if tracing {
            trace_entries.push(TraceEntry {
                phase: "dialect_lower".to_string(),
                duration: dialect_start.elapsed(),
                detail: Some(format!("{} rewrites", rewrites)),
            });
        }

        // Phase 1: Translate tMIR module to internal LIR functions.
        let adapter_start = Instant::now();
        let lir_functions = llvm2_lower::translate_module(&lowered_module)?;
        if tracing {
            trace_entries.push(TraceEntry {
                phase: "adapter".to_string(),
                duration: adapter_start.elapsed(),
                detail: Some(format!("{} functions", lir_functions.len())),
            });
        }

        if lir_functions.is_empty() {
            return Err(CompileError::EmptyModule);
        }

        // Target dispatch (#340): route to the per-target backend based on
        // `config.target`. AArch64 is the default and fully wired; x86-64 is
        // dispatched to the parallel `X86Pipeline`. Other targets are not
        // yet wired.
        match self.config.target {
            Target::Aarch64 => {}
            Target::X86_64 => {
                return self.compile_x86_64(lir_functions, total_start, tracing, trace_entries);
            }
            Target::Riscv64 => {
                return Err(CompileError::Pipeline(PipelineError::ISel(
                    "RISC-V target dispatch not yet wired in Compiler::compile; \
                     use llvm2_codegen::riscv APIs directly for now"
                        .to_string(),
                )));
            }
        }

        // Build the internal pipeline.
        let pipeline = self.build_pipeline();

        // Phase 2+: Prepare each function through ISel, optimization,
        // regalloc, frame lowering, and branch resolution. All functions
        // are then combined into a single Mach-O .o via compile_module()
        // so cross-function BL instructions get proper BRANCH26 relocations.
        //
        // When parallel compilation is enabled and there are 2+ functions,
        // use rayon to prepare functions concurrently. Each function's
        // pipeline (ISel -> opt -> regalloc -> frame -> branch resolution)
        // is fully independent with no shared mutable state.
        let use_parallel = self.config.parallel && lir_functions.len() >= 2;

        let mut prepared_funcs: Vec<llvm2_ir::MachFunction>;

        if use_parallel {
            // Parallel path: each function is prepared independently via rayon.
            // Collect results with optional trace entries, then unpack.
            let results: Vec<Result<(llvm2_ir::MachFunction, Option<TraceEntry>), CompileError>> =
                lir_functions
                    .par_iter()
                    .map(|(lir_func, proof_ctx)| {
                        let func_start = Instant::now();
                        let ir_func = pipeline
                            .prepare_function_with_proofs(lir_func, Some(proof_ctx))
                            .map_err(CompileError::Pipeline)?;
                        let entry = if tracing {
                            Some(TraceEntry {
                                phase: "prepare_function".to_string(),
                                duration: func_start.elapsed(),
                                detail: Some(ir_func.name.clone()),
                            })
                        } else {
                            None
                        };
                        Ok((ir_func, entry))
                    })
                    .collect();

            prepared_funcs = Vec::with_capacity(results.len());
            for result in results {
                let (ir_func, trace_entry) = result?;
                if let Some(entry) = trace_entry {
                    trace_entries.push(entry);
                }
                prepared_funcs.push(ir_func);
            }
        } else {
            // Sequential path: single function or parallel disabled.
            prepared_funcs = Vec::with_capacity(lir_functions.len());
            for (lir_func, proof_ctx) in &lir_functions {
                let func_start = Instant::now();

                let ir_func = pipeline.prepare_function_with_proofs(lir_func, Some(proof_ctx))?;

                if tracing {
                    trace_entries.push(TraceEntry {
                        phase: "prepare_function".to_string(),
                        duration: func_start.elapsed(),
                        detail: Some(ir_func.name.clone()),
                    });
                }

                prepared_funcs.push(ir_func);
            }
        }

        let function_count = prepared_funcs.len();

        // Phase 8-9: Encode all functions and emit a single Mach-O .o file
        // with proper cross-function BRANCH26 relocations.
        // When parallel mode is active, use parallel encoding to avoid the
        // sequential bottleneck of encoding functions one-by-one.
        let module_start = Instant::now();
        let obj_bytes = if use_parallel {
            pipeline.compile_module_parallel(&prepared_funcs)?
        } else {
            pipeline.compile_module(&prepared_funcs)?
        };

        if tracing {
            trace_entries.push(TraceEntry {
                phase: "compile_module".to_string(),
                duration: module_start.elapsed(),
                detail: Some(format!("{} functions", function_count)),
            });
        }

        // Count actual non-pseudo instructions across all prepared functions.
        // Each AArch64 instruction is exactly 4 bytes, so code_size = count * 4.
        // This is the real instruction count, not the Mach-O object size / 4
        // which would incorrectly include headers, symbol tables, and relocations.
        let total_instruction_count: usize = prepared_funcs
            .iter()
            .map(|f| count_real_instructions(f))
            .sum();
        let total_code_size = total_instruction_count * 4;

        // Query actual pass count from the optimization pipeline rather than
        // using hardcoded estimates (fixes #272).
        let opt_passes_per_func = {
            use llvm2_opt::pipeline::{OptLevel as OptOptLevel, OptimizationPipeline};
            let opt_level = match self.config.opt_level {
                OptLevel::O0 => OptOptLevel::O0,
                OptLevel::O1 => OptOptLevel::O1,
                OptLevel::O2 => OptOptLevel::O2,
                OptLevel::O3 => OptOptLevel::O3,
            };
            OptimizationPipeline::new(opt_level).pass_count()
        };

        let metrics = CompilationMetrics {
            code_size_bytes: total_code_size,
            instruction_count: total_instruction_count,
            function_count,
            optimization_passes_run: opt_passes_per_func * function_count,
        };

        let trace = if tracing {
            Some(CompilerTrace {
                entries: trace_entries,
                total_duration: total_start.elapsed(),
            })
        } else {
            None
        };

        // Proof certificates: run llvm2-verify function verifier on each
        // prepared function to produce per-instruction proof certificates.
        // When parallel compilation is enabled, generate certificates
        // concurrently since each function's verification is independent.
        // Requires the `verify` feature; without it, proofs are always None.
        #[cfg(feature = "verify")]
        let proofs = if self.config.emit_proofs {
            let all_certs: Vec<ProofCertificate> = if use_parallel {
                prepared_funcs
                    .par_iter()
                    .flat_map(|func| generate_proof_certificates(func))
                    .collect()
            } else {
                let mut certs = Vec::new();
                for func in &prepared_funcs {
                    certs.extend(generate_proof_certificates(func));
                }
                certs
            };
            Some(all_certs)
        } else {
            None
        };
        #[cfg(not(feature = "verify"))]
        let proofs: Option<Vec<ProofCertificate>> = None;

        Ok(CompilationResult {
            object_code: obj_bytes,
            metrics,
            trace,
            proofs,
        })
    }

    /// x86-64 compile path (#340, #464).
    ///
    /// Routes tMIR functions through the [`crate::x86_64::X86Pipeline`]
    /// (ISel -> regalloc -> frame lowering -> encoding) and then emits a
    /// single multi-function Mach-O / ELF object via
    /// [`crate::x86_64::X86Pipeline::compile_module`]. Cross-function calls
    /// are wired via symbol-table entries and `X86_64_RELOC_BRANCH` /
    /// `R_X86_64_PLT32` relocations.
    ///
    /// Mirrors the AArch64 dispatcher at
    /// [`Self::compile_aarch64`]: each function is run through ISel
    /// independently, then the entire set is handed to `compile_module` for
    /// combined emission.
    ///
    /// Note: `CompilationMetrics::instruction_count` is reported from the
    /// x86-64 ISel functions (non-pseudo insts summed across all functions),
    /// and `code_size_bytes` is the raw encoded code length (no Mach-O
    /// header/symbol-table overhead). Because x86-64 uses variable-length
    /// encoding, `code_size_bytes` is NOT `instruction_count * 4`.
    fn compile_x86_64(
        &self,
        lir_functions: Vec<(llvm2_lower::Function, llvm2_lower::ProofContext)>,
        total_start: Instant,
        tracing: bool,
        mut trace_entries: Vec<TraceEntry>,
    ) -> Result<CompilationResult, CompileError> {
        use crate::x86_64::{X86OutputFormat, X86Pipeline, X86PipelineConfig};

        // #465: x86-64 proof certificates are now produced via the shared
        // public `ProofCertificate` shape. The previous early-return guard
        // (`CompileError::ProofsUnsupportedForTarget`) was removed once
        // `all_x86_64_proofs` (#434) and `all_x86_64_eflags_proofs` (#458)
        // landed in the `ProofDatabase` and a parallel x86-64 function
        // verifier (`llvm2_verify::x86_64_function_verifier`) was wired to
        // walk an `X86ISelFunction`. Proof emission is gated on the
        // `verify` feature flag, same as the AArch64 path. When the
        // feature is off, `emit_proofs` is honored as a no-op and the
        // returned `proofs` field is `None`.
        //
        // Without the `verify` feature the `verify_x86_64_function` call
        // below is unreachable, so we still surface a typed error if a
        // caller opts into proofs on a non-verify build — otherwise the
        // invariant "result.proofs.is_some() when emit_proofs=true" would
        // silently regress.
        #[cfg(not(feature = "verify"))]
        if self.config.emit_proofs {
            return Err(CompileError::ProofsUnsupportedForTarget {
                target: Target::X86_64,
            });
        }

        let function_count = lir_functions.len();
        if function_count == 0 {
            return Err(CompileError::Pipeline(PipelineError::ISel(
                "x86_64 dispatcher received an empty module".to_string(),
            )));
        }

        // Phase 1: Run ISel per function. Each X86ISelFunction is
        // independent (no shared mutable state), so this is straightforward
        // sequential work; parallelization is a follow-up (mirrors the
        // AArch64 `prepared_funcs` vector).
        let isel_start = Instant::now();
        let mut isel_funcs: Vec<llvm2_lower::x86_64_isel::X86ISelFunction> =
            Vec::with_capacity(function_count);
        for (lir_func, _proof_ctx) in &lir_functions {
            use llvm2_lower::x86_64_isel::X86InstructionSelector;
            let sig = llvm2_lower::function::Signature {
                params: lir_func.signature.params.clone(),
                returns: lir_func.signature.returns.clone(),
            };
            let mut isel = X86InstructionSelector::new(lir_func.name.clone(), sig.clone());
            isel.seed_value_types(&lir_func.value_types);
            isel.lower_formal_arguments(&sig, lir_func.entry_block)
                .map_err(|e| CompileError::Pipeline(PipelineError::ISel(e.to_string())))?;
            let mut block_order: Vec<_> = lir_func.blocks.keys().copied().collect();
            block_order.sort_by_key(|b| b.0);
            for block_ref in &block_order {
                let basic_block = &lir_func.blocks[block_ref];
                isel.select_block(*block_ref, &basic_block.instructions)
                    .map_err(|e| CompileError::Pipeline(PipelineError::ISel(e.to_string())))?;
            }
            isel_funcs.push(isel.finalize());
        }

        if tracing {
            trace_entries.push(TraceEntry {
                phase: "x86_isel".to_string(),
                duration: isel_start.elapsed(),
                detail: Some(format!("{} functions", function_count)),
            });
        }

        // Count non-pseudo ISel instructions across all functions for metrics.
        let instruction_count: usize = isel_funcs
            .iter()
            .flat_map(|f| {
                f.block_order
                    .iter()
                    .filter_map(move |blk| f.blocks.get(blk))
                    .flat_map(|b| b.insts.iter())
            })
            .filter(|i| !i.opcode.is_pseudo())
            .count();

        // Raw code size: encode each function with RawBytes output and sum.
        // Compile_module with RawBytes returns the concatenated per-function
        // code (with inline const pools) — this is the raw code size,
        // excluding any Mach-O / ELF object wrapper overhead.
        let raw_bytes = {
            let pipeline = X86Pipeline::new(X86PipelineConfig {
                output_format: X86OutputFormat::RawBytes,
                emit_frame: true,
                ..X86PipelineConfig::default()
            });
            pipeline
                .compile_module(&isel_funcs)
                .map_err(|e| CompileError::Pipeline(PipelineError::ISel(e.to_string())))?
        };
        let total_code_size = raw_bytes.len();

        // Final Mach-O object bytes via compile_module. This is the object
        // returned to the caller; cross-function CALL fixups become
        // X86_64_RELOC_BRANCH relocations.
        let encode_start = Instant::now();
        let obj_bytes = {
            let pipeline = X86Pipeline::new(X86PipelineConfig {
                output_format: X86OutputFormat::MachO,
                emit_frame: true,
                ..X86PipelineConfig::default()
            });
            pipeline
                .compile_module(&isel_funcs)
                .map_err(|e| CompileError::Pipeline(PipelineError::ISel(e.to_string())))?
        };
        if tracing {
            trace_entries.push(TraceEntry {
                phase: "x86_compile_module".to_string(),
                duration: encode_start.elapsed(),
                detail: Some(format!("{} functions", function_count)),
            });
        }

        // Optimization pass count for metrics parity with the AArch64 path.
        // Note: x86_64 optimization passes are not yet wired into the
        // X86Pipeline (tracked separately as #438), so this is the *target*
        // pass count at the given opt level — not the number actually run.
        let opt_passes_per_func = {
            use llvm2_opt::pipeline::{OptLevel as OptOptLevel, OptimizationPipeline};
            let opt_level = match self.config.opt_level {
                OptLevel::O0 => OptOptLevel::O0,
                OptLevel::O1 => OptOptLevel::O1,
                OptLevel::O2 => OptOptLevel::O2,
                OptLevel::O3 => OptOptLevel::O3,
            };
            OptimizationPipeline::new(opt_level).pass_count()
        };

        let metrics = CompilationMetrics {
            code_size_bytes: total_code_size,
            instruction_count,
            function_count,
            optimization_passes_run: opt_passes_per_func * function_count,
        };

        let trace = if tracing {
            Some(CompilerTrace {
                entries: trace_entries,
                total_duration: total_start.elapsed(),
            })
        } else {
            None
        };

        // Proof certificates: the x86-64 pipeline does not yet produce
        // per-instruction proof certificates (#465).
        //
        // Mirrors the AArch64 path in `Compiler::compile`: each
        // X86ISelFunction is independent, so generation runs sequentially
        // here (rayon parallelization is a straightforward follow-up).
        // Emission is gated on the `verify` feature flag, same as the
        // AArch64 path. When the feature is off, the `emit_proofs = true`
        // case was already rejected above with
        // `CompileError::ProofsUnsupportedForTarget`, preserving the
        // invariant that `result.proofs.is_some()` whenever
        // `config.emit_proofs == true`.
        #[cfg(feature = "verify")]
        let proofs = if self.config.emit_proofs {
            let mut all_certs: Vec<ProofCertificate> = Vec::new();
            for func in &isel_funcs {
                all_certs.extend(generate_x86_64_proof_certificates(func));
            }
            Some(all_certs)
        } else {
            None
        };
        #[cfg(not(feature = "verify"))]
        let proofs: Option<Vec<ProofCertificate>> = None;

        Ok(CompilationResult {
            object_code: obj_bytes,
            metrics,
            trace,
            proofs,
        })
    }

    /// Compile a pre-built IR function (skipping tMIR adapter and ISel).
    ///
    /// Useful when you already have an `IrMachFunction` and want to run
    /// optimization, regalloc, frame lowering, encoding, and Mach-O emission.
    pub fn compile_ir_function(
        &self,
        ir_func: &mut llvm2_ir::MachFunction,
    ) -> Result<CompilationResult, CompileError> {
        let start = Instant::now();
        let pipeline = self.build_pipeline();

        let obj_bytes = pipeline.compile_ir_function(ir_func)?;

        // Count actual non-pseudo instructions from the function, not obj size / 4
        // which would include Mach-O headers, symbol tables, and relocations.
        let code_insts = count_real_instructions(ir_func);
        // Query actual pass count from the optimization pipeline (fixes #272).
        let opt_passes = {
            use llvm2_opt::pipeline::{OptLevel as OptOptLevel, OptimizationPipeline};
            let opt_level = match self.config.opt_level {
                OptLevel::O0 => OptOptLevel::O0,
                OptLevel::O1 => OptOptLevel::O1,
                OptLevel::O2 => OptOptLevel::O2,
                OptLevel::O3 => OptOptLevel::O3,
            };
            OptimizationPipeline::new(opt_level).pass_count()
        };

        let metrics = CompilationMetrics {
            code_size_bytes: code_insts * 4,
            instruction_count: code_insts,
            function_count: 1,
            optimization_passes_run: opt_passes,
        };

        let trace = if self.config.trace_level != CompilerTraceLevel::None {
            Some(CompilerTrace {
                entries: vec![TraceEntry {
                    phase: "compile_ir_function".to_string(),
                    duration: start.elapsed(),
                    detail: Some(ir_func.name.clone()),
                }],
                total_duration: start.elapsed(),
            })
        } else {
            None
        };

        #[cfg(feature = "verify")]
        let proofs = if self.config.emit_proofs {
            Some(generate_proof_certificates(ir_func))
        } else {
            None
        };
        #[cfg(not(feature = "verify"))]
        let proofs: Option<Vec<ProofCertificate>> = None;

        Ok(CompilationResult {
            object_code: obj_bytes,
            metrics,
            trace,
            proofs,
        })
    }

    /// Compile a tMIR module to executable memory for JIT execution.
    ///
    /// Translates each function in the module through the full pipeline:
    /// tMIR adapter -> ISel -> optimization -> regalloc -> frame lowering
    /// -> branch resolution -> JIT linking.
    ///
    /// All functions are compiled into a single executable buffer with each
    /// function as a separate symbol. Cross-function branches are resolved
    /// directly in memory and external symbols are bound from `extern_symbols`.
    ///
    /// Returns the executable buffer, metrics, optional trace, and optional
    /// proof certificates.
    pub fn compile_module_to_jit(
        &self,
        module: &tmir::Module,
        extern_symbols: &std::collections::HashMap<String, *const u8>,
    ) -> Result<JitCompilationResult, CompileError> {
        let total_start = Instant::now();
        let tracing = self.config.trace_level != CompilerTraceLevel::None;
        let mut trace_entries = Vec::new();

        // Phase 0: Pre-adapter dialect lowering (#433, tMIR #428).
        //
        // Runs `tmir::dialect::lower_module` with an internal
        // `DialectRegistry` so any `Inst::DialectOp` (e.g. `verif.bfs_step`,
        // `verif.frontier_drain`) is rewritten into core tMIR before the
        // adapter runs. Unknown dialects are rejected here — the adapter
        // has no DialectOp handler and would otherwise fail at ISel.
        //
        // Clones the module locally because the dialect driver needs
        // `&mut Module` and the public `compile` signature is `&Module`.
        // The clone is O(module) and only traverses the IR once; if this
        // ever shows up in a profile we can lift the path to a
        // `compile_owned` API instead of cloning.
        let dialect_start = Instant::now();
        let mut lowered_module = module.clone();
        let rewrites = lower_dialects(&mut lowered_module)?;
        if tracing {
            trace_entries.push(TraceEntry {
                phase: "dialect_lower".to_string(),
                duration: dialect_start.elapsed(),
                detail: Some(format!("{} rewrites", rewrites)),
            });
        }

        // Phase 1: Translate tMIR module to internal LIR functions.
        let adapter_start = Instant::now();
        let lir_functions = llvm2_lower::translate_module(&lowered_module)?;
        if tracing {
            trace_entries.push(TraceEntry {
                phase: "adapter".to_string(),
                duration: adapter_start.elapsed(),
                detail: Some(format!("{} functions", lir_functions.len())),
            });
        }

        if lir_functions.is_empty() {
            return Err(CompileError::EmptyModule);
        }

        // Build the internal pipeline.
        let pipeline = self.build_pipeline();

        // Phase 2+: Prepare each function through ISel, optimization,
        // regalloc, frame lowering, and branch resolution. The prepared
        // functions are then handed to the JIT compiler for in-memory code
        // generation and cross-function fixup resolution.
        //
        // When parallel compilation is enabled and there are 2+ functions,
        // use rayon to prepare functions concurrently. Each function's
        // pipeline (ISel -> opt -> regalloc -> frame -> branch resolution)
        // is fully independent with no shared mutable state.
        let use_parallel = self.config.parallel && lir_functions.len() >= 2;

        let mut prepared_funcs: Vec<llvm2_ir::MachFunction>;
        let mut preparation_metrics: Vec<crate::pipeline::PreparationMetrics>;

        if use_parallel {
            // Parallel path: each function is prepared independently via rayon.
            // Collect results with optional trace entries, then unpack.
            let results: Vec<
                Result<
                    (
                        llvm2_ir::MachFunction,
                        crate::pipeline::PreparationMetrics,
                        Option<TraceEntry>,
                    ),
                    CompileError,
                >,
            > = lir_functions
                .par_iter()
                .map(|(lir_func, proof_ctx)| {
                    let func_start = Instant::now();
                    let (ir_func, metrics) = pipeline
                        .prepare_function_with_metrics(lir_func, Some(proof_ctx))
                        .map_err(CompileError::Pipeline)?;
                    let entry = if tracing {
                        Some(TraceEntry {
                            phase: "prepare_function".to_string(),
                            duration: func_start.elapsed(),
                            detail: Some(ir_func.name.clone()),
                        })
                    } else {
                        None
                    };
                    Ok((ir_func, metrics, entry))
                })
                .collect();

            prepared_funcs = Vec::with_capacity(results.len());
            preparation_metrics = Vec::with_capacity(results.len());
            for result in results {
                let (ir_func, metrics, trace_entry) = result?;
                if let Some(entry) = trace_entry {
                    trace_entries.push(entry);
                }
                preparation_metrics.push(metrics);
                prepared_funcs.push(ir_func);
            }
        } else {
            // Sequential path: single function or parallel disabled.
            prepared_funcs = Vec::with_capacity(lir_functions.len());
            preparation_metrics = Vec::with_capacity(lir_functions.len());
            for (lir_func, proof_ctx) in &lir_functions {
                let func_start = Instant::now();

                let (ir_func, metrics) =
                    pipeline.prepare_function_with_metrics(lir_func, Some(proof_ctx))?;

                if tracing {
                    trace_entries.push(TraceEntry {
                        phase: "prepare_function".to_string(),
                        duration: func_start.elapsed(),
                        detail: Some(ir_func.name.clone()),
                    });
                }

                preparation_metrics.push(metrics);
                prepared_funcs.push(ir_func);
            }
        }

        let function_count = prepared_funcs.len();

        // Final phase: encode all prepared functions into executable memory,
        // resolve internal cross-function branches, and bind external symbols.
        let jit_start = Instant::now();
        let jit = crate::jit::JitCompiler::new(crate::jit::JitConfig {
            opt_level: self.config.opt_level,
            verify: false,
            // #375: Inherit the JitConfig default (ErrorOnFailure) so
            // dispatch verification failures surface here too rather than
            // being silently rewritten to a CPU-only plan. compile_raw does
            // not currently invoke the dispatch verifier, so this is mostly
            // defensive wiring for future heterogeneous-aware code paths.
            ..Default::default()
        });
        let (buffer, encoding_timings) = jit
            .compile_raw_with_encoding_metrics(&prepared_funcs, extern_symbols)
            .map_err(CompileError::Jit)?;

        if tracing {
            trace_entries.push(TraceEntry {
                phase: "compile_raw".to_string(),
                duration: jit_start.elapsed(),
                detail: Some(format!("{} functions", function_count)),
            });
        }

        let per_function_metrics: Vec<FunctionQualityMetrics> = prepared_funcs
            .iter()
            .zip(preparation_metrics.into_iter())
            .map(|(func, prep_metrics)| {
                let mut phase_timings = prep_metrics.timings;
                phase_timings.encoding = encoding_timings.get(func.name.as_str()).copied();
                FunctionQualityMetrics {
                    name: func.name.clone(),
                    instruction_count: count_real_instructions(func),
                    spill_slot_count: prep_metrics.spill_slot_count,
                    branch_count: count_branch_instructions(func),
                    phase_timings,
                }
            })
            .collect();

        // Count actual non-pseudo instructions across all prepared functions.
        // Each AArch64 instruction is exactly 4 bytes, so code_size = count * 4.
        // This is the real instruction count, not the allocated JIT buffer size
        // which may include page alignment and veneer trampolines.
        let total_instruction_count: usize = prepared_funcs
            .iter()
            .map(|f| count_real_instructions(f))
            .sum();
        let total_code_size = total_instruction_count * 4;

        // Query actual pass count from the optimization pipeline rather than
        // using hardcoded estimates (fixes #272).
        let opt_passes_per_func = {
            use llvm2_opt::pipeline::{OptLevel as OptOptLevel, OptimizationPipeline};
            let opt_level = match self.config.opt_level {
                OptLevel::O0 => OptOptLevel::O0,
                OptLevel::O1 => OptOptLevel::O1,
                OptLevel::O2 => OptOptLevel::O2,
                OptLevel::O3 => OptOptLevel::O3,
            };
            OptimizationPipeline::new(opt_level).pass_count()
        };

        let metrics = CompilationMetrics {
            code_size_bytes: total_code_size,
            instruction_count: total_instruction_count,
            function_count,
            optimization_passes_run: opt_passes_per_func * function_count,
        };

        let trace = if tracing {
            Some(CompilerTrace {
                entries: trace_entries,
                total_duration: total_start.elapsed(),
            })
        } else {
            None
        };

        // Proof certificates: run llvm2-verify function verifier on each
        // prepared function to produce per-instruction proof certificates.
        // When parallel compilation is enabled, generate certificates
        // concurrently since each function's verification is independent.
        // Requires the `verify` feature; without it, proofs are always None.
        #[cfg(feature = "verify")]
        let proofs = if self.config.emit_proofs {
            let all_certs: Vec<ProofCertificate> = if use_parallel {
                prepared_funcs
                    .par_iter()
                    .flat_map(|func| generate_proof_certificates(func))
                    .collect()
            } else {
                let mut certs = Vec::new();
                for func in &prepared_funcs {
                    certs.extend(generate_proof_certificates(func));
                }
                certs
            };
            Some(all_certs)
        } else {
            None
        };
        #[cfg(not(feature = "verify"))]
        let proofs: Option<Vec<ProofCertificate>> = None;

        Ok(JitCompilationResult {
            buffer,
            metrics,
            trace,
            proofs,
            per_function_metrics,
        })
    }

    /// Build the internal [`Pipeline`] from the compiler configuration.
    fn build_pipeline(&self) -> Pipeline {
        Pipeline::new(PipelineConfig {
            opt_level: self.config.opt_level,
            emit_debug: self.config.emit_debug,
            verify_dispatch: crate::pipeline::DispatchVerifyMode::FallbackOnFailure,
            verify: false,
            enable_post_ra_opt: self.config.opt_level != crate::pipeline::OptLevel::O0,
            use_pressure_aware_scheduler: matches!(
                self.config.opt_level,
                crate::pipeline::OptLevel::O2 | crate::pipeline::OptLevel::O3
            ),
            cegis_superopt_budget_sec: self.config.cegis_superopt_budget_sec,
            target_triple: format!("{}-unknown-unknown", self.config.target.name()),
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    // Exercises the deprecated `ExecutableBuffer::get_fn_ptr` API as part of
    // regression coverage for JIT symbol enumeration. The deprecation is
    // tracked by issue #355 and migration is out of scope for this module.
    #![allow(deprecated)]
    use super::*;

    #[test]
    fn test_default_config() {
        let config = CompilerConfig::default();
        assert_eq!(config.opt_level, OptLevel::O2);
        assert_eq!(config.target, Target::Aarch64);
        assert!(!config.emit_proofs);
        assert_eq!(config.trace_level, CompilerTraceLevel::None);
        assert!(
            config.parallel,
            "parallel compilation should be enabled by default"
        );
    }

    #[test]
    fn test_compiler_new() {
        let compiler = Compiler::new(CompilerConfig::default());
        assert_eq!(compiler.config().opt_level, OptLevel::O2);
        assert_eq!(compiler.config().target, Target::Aarch64);
    }

    #[test]
    fn test_compiler_default_o2() {
        let compiler = Compiler::default_o2();
        assert_eq!(compiler.config().opt_level, OptLevel::O2);
    }

    #[test]
    fn test_custom_config() {
        let config = CompilerConfig {
            opt_level: OptLevel::O0,
            target: Target::X86_64,
            emit_proofs: true,
            trace_level: CompilerTraceLevel::Full,
            emit_debug: true,
            parallel: false,
            cegis_superopt_budget_sec: None,
        };
        let compiler = Compiler::new(config);
        assert_eq!(compiler.config().opt_level, OptLevel::O0);
        assert_eq!(compiler.config().target, Target::X86_64);
        assert!(compiler.config().emit_proofs);
        assert_eq!(compiler.config().trace_level, CompilerTraceLevel::Full);
        assert!(compiler.config().emit_debug);
        assert!(!compiler.config().parallel);
    }

    #[test]
    fn test_metrics_default() {
        let metrics = CompilationMetrics::default();
        assert_eq!(metrics.code_size_bytes, 0);
        assert_eq!(metrics.instruction_count, 0);
        assert_eq!(metrics.function_count, 0);
        assert_eq!(metrics.optimization_passes_run, 0);
    }

    #[test]
    fn test_compiler_trace_default() {
        let trace = CompilerTrace::default();
        assert!(trace.entries.is_empty());
        assert_eq!(trace.total_duration, Duration::ZERO);
    }

    #[test]
    fn test_compile_ir_function_add() {
        // Build a simple add function in IR (post-ISel, post-regalloc state)
        // and compile it through the pipeline.
        let mut ir_func = crate::pipeline::build_add_test_function();
        let compiler = Compiler::default_o2();
        let result = compiler.compile_ir_function(&mut ir_func).unwrap();

        assert!(
            !result.object_code.is_empty(),
            "should produce Mach-O bytes"
        );
        assert!(result.metrics.code_size_bytes > 0);
        assert_eq!(result.metrics.function_count, 1);
        assert!(
            result.trace.is_none(),
            "trace should be None with default config"
        );
        assert!(
            result.proofs.is_none(),
            "proofs should be None with default config"
        );
    }

    #[test]
    fn test_compile_ir_function_with_trace() {
        let mut ir_func = crate::pipeline::build_add_test_function();
        let compiler = Compiler::new(CompilerConfig {
            trace_level: CompilerTraceLevel::Summary,
            ..CompilerConfig::default()
        });
        let result = compiler.compile_ir_function(&mut ir_func).unwrap();

        assert!(
            result.trace.is_some(),
            "trace should be present with Summary level"
        );
        let trace = result.trace.unwrap();
        assert!(!trace.entries.is_empty());
        assert!(trace.total_duration >= Duration::ZERO);
    }

    #[test]
    fn test_compile_ir_function_with_proofs() {
        let mut ir_func = crate::pipeline::build_add_test_function();
        let compiler = Compiler::new(CompilerConfig {
            emit_proofs: true,
            ..CompilerConfig::default()
        });
        let result = compiler.compile_ir_function(&mut ir_func).unwrap();

        assert!(
            result.proofs.is_some(),
            "proofs field should be Some when emit_proofs is true"
        );
        let proofs = result.proofs.unwrap();
        // The add test function contains an AddRR instruction which has a
        // verified proof obligation in the proof database.
        assert!(
            !proofs.is_empty(),
            "proof certificates should be non-empty for add function"
        );
        // Every certificate that was emitted should be verified (no failures).
        for cert in &proofs {
            assert!(
                cert.verified,
                "certificate '{}' should be verified",
                cert.rule_name
            );
            assert!(!cert.rule_name.is_empty(), "rule_name should not be empty");
            assert!(!cert.category.is_empty(), "category should not be empty");
            assert!(!cert.strength.is_empty(), "strength should not be empty");
            assert!(
                !cert.function_name.is_empty(),
                "function_name should not be empty"
            );
        }
    }

    #[test]
    fn test_compiler_trace_level_equality() {
        assert_eq!(CompilerTraceLevel::None, CompilerTraceLevel::None);
        assert_ne!(CompilerTraceLevel::None, CompilerTraceLevel::Summary);
        assert_ne!(CompilerTraceLevel::Summary, CompilerTraceLevel::Full);
    }

    #[test]
    fn test_parallel_and_serial_ir_compile_produce_same_output() {
        // Compile the same IR function with parallel=true and parallel=false.
        // Single-function modules take the sequential path regardless of the
        // parallel flag (threshold is 2+ functions), so this test verifies
        // the config plumbing and that both paths produce identical output.
        let mut ir_func_a = crate::pipeline::build_add_test_function();
        let mut ir_func_b = crate::pipeline::build_add_test_function();

        let serial = Compiler::new(CompilerConfig {
            parallel: false,
            ..CompilerConfig::default()
        });
        let parallel = Compiler::new(CompilerConfig {
            parallel: true,
            ..CompilerConfig::default()
        });

        let result_serial = serial.compile_ir_function(&mut ir_func_a).unwrap();
        let result_parallel = parallel.compile_ir_function(&mut ir_func_b).unwrap();

        assert_eq!(
            result_serial.object_code, result_parallel.object_code,
            "parallel and serial compilation must produce identical Mach-O output"
        );
        assert_eq!(
            result_serial.metrics.instruction_count,
            result_parallel.metrics.instruction_count
        );
    }

    #[test]
    fn test_parallel_config_disabled() {
        let config = CompilerConfig {
            parallel: false,
            ..CompilerConfig::default()
        };
        assert!(!config.parallel);
    }

    #[test]
    fn test_parallel_multi_function_module_produces_same_output() {
        // Build a tMIR module with 3 functions and compile it both serially
        // and in parallel. The outputs MUST be identical (deterministic).
        use tmir::Ty;
        use tmir_build::ModuleBuilder;

        let build_module = || {
            let mut mb = ModuleBuilder::new("test_parallel_multi");
            // Function 1: f(a, b) = a + b
            {
                let ty = mb.add_func_type(vec![Ty::I64, Ty::I64], vec![Ty::I64]);
                let mut fb = mb.function("add_fn", ty);
                let entry = fb.create_block();
                let a = fb.add_block_param(entry, Ty::I64);
                let b = fb.add_block_param(entry, Ty::I64);
                fb.switch_to_block(entry);
                let result = fb.add(Ty::I64, a, b);
                fb.ret(vec![result]);
                fb.build();
            }
            // Function 2: g(a, b) = a * b - a
            {
                let ty = mb.add_func_type(vec![Ty::I64, Ty::I64], vec![Ty::I64]);
                let mut fb = mb.function("mul_sub_fn", ty);
                let entry = fb.create_block();
                let a = fb.add_block_param(entry, Ty::I64);
                let b = fb.add_block_param(entry, Ty::I64);
                fb.switch_to_block(entry);
                let prod = fb.mul(Ty::I64, a, b);
                let result = fb.sub(Ty::I64, prod, a);
                fb.ret(vec![result]);
                fb.build();
            }
            // Function 3: h(a, b) = (a + b) * 2
            {
                let ty = mb.add_func_type(vec![Ty::I64, Ty::I64], vec![Ty::I64]);
                let mut fb = mb.function("add_double_fn", ty);
                let entry = fb.create_block();
                let a = fb.add_block_param(entry, Ty::I64);
                let b = fb.add_block_param(entry, Ty::I64);
                fb.switch_to_block(entry);
                let sum = fb.add(Ty::I64, a, b);
                let c2 = fb.iconst(Ty::I64, 2);
                let result = fb.mul(Ty::I64, sum, c2);
                fb.ret(vec![result]);
                fb.build();
            }
            mb.build()
        };

        let module_serial = build_module();
        let module_parallel = build_module();

        let serial = Compiler::new(CompilerConfig {
            parallel: false,
            ..CompilerConfig::default()
        });
        let parallel = Compiler::new(CompilerConfig {
            parallel: true,
            ..CompilerConfig::default()
        });

        let result_serial = serial.compile(&module_serial).unwrap();
        let result_parallel = parallel.compile(&module_parallel).unwrap();

        assert_eq!(
            result_serial.object_code, result_parallel.object_code,
            "multi-function parallel and serial compilation must produce identical Mach-O output"
        );
        assert_eq!(result_serial.metrics.function_count, 3);
        assert_eq!(result_parallel.metrics.function_count, 3);
        assert_eq!(
            result_serial.metrics.instruction_count,
            result_parallel.metrics.instruction_count
        );
    }

    // -----------------------------------------------------------------------
    // JIT batch compilation tests
    // -----------------------------------------------------------------------

    /// Compile a single-function tMIR module to JIT and verify the symbol
    /// is present in the executable buffer.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_compile_module_to_jit_single_function() {
        use std::collections::HashMap;
        use tmir::Ty;
        use tmir_build::ModuleBuilder;

        let mut mb = ModuleBuilder::new("jit_single");
        {
            let ty = mb.add_func_type(vec![Ty::I64, Ty::I64], vec![Ty::I64]);
            let mut fb = mb.function("add_fn", ty);
            let entry = fb.create_block();
            let a = fb.add_block_param(entry, Ty::I64);
            let b = fb.add_block_param(entry, Ty::I64);
            fb.switch_to_block(entry);
            let result = fb.add(Ty::I64, a, b);
            fb.ret(vec![result]);
            fb.build();
        }
        let module = mb.build();

        let compiler = Compiler::default_o2();
        let result = compiler
            .compile_module_to_jit(&module, &HashMap::new())
            .unwrap();

        assert_eq!(result.metrics.function_count, 1);
        assert!(result.metrics.instruction_count > 0);
        assert!(result.metrics.code_size_bytes > 0);
        assert!(
            result.buffer.get_fn_ptr("add_fn").is_some(),
            "add_fn should be in the symbol map"
        );
        assert_eq!(result.buffer.symbol_count(), 1);
    }

    /// Compile a multi-function tMIR module to JIT and verify all symbols
    /// are present with correct metrics.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_compile_module_to_jit_multi_function() {
        use std::collections::HashMap;
        use tmir::Ty;
        use tmir_build::ModuleBuilder;

        let mut mb = ModuleBuilder::new("jit_multi");
        {
            let ty = mb.add_func_type(vec![Ty::I64, Ty::I64], vec![Ty::I64]);
            let mut fb = mb.function("add_fn", ty);
            let entry = fb.create_block();
            let a = fb.add_block_param(entry, Ty::I64);
            let b = fb.add_block_param(entry, Ty::I64);
            fb.switch_to_block(entry);
            let result = fb.add(Ty::I64, a, b);
            fb.ret(vec![result]);
            fb.build();
        }
        {
            let ty = mb.add_func_type(vec![Ty::I64, Ty::I64], vec![Ty::I64]);
            let mut fb = mb.function("mul_fn", ty);
            let entry = fb.create_block();
            let a = fb.add_block_param(entry, Ty::I64);
            let b = fb.add_block_param(entry, Ty::I64);
            fb.switch_to_block(entry);
            let result = fb.mul(Ty::I64, a, b);
            fb.ret(vec![result]);
            fb.build();
        }
        {
            let ty = mb.add_func_type(vec![Ty::I64, Ty::I64], vec![Ty::I64]);
            let mut fb = mb.function("sub_fn", ty);
            let entry = fb.create_block();
            let a = fb.add_block_param(entry, Ty::I64);
            let b = fb.add_block_param(entry, Ty::I64);
            fb.switch_to_block(entry);
            let result = fb.sub(Ty::I64, a, b);
            fb.ret(vec![result]);
            fb.build();
        }
        let module = mb.build();

        let compiler = Compiler::default_o2();
        let result = compiler
            .compile_module_to_jit(&module, &HashMap::new())
            .unwrap();

        assert_eq!(result.metrics.function_count, 3);
        assert!(result.metrics.instruction_count > 0);
        assert_eq!(result.buffer.symbol_count(), 3);
        assert!(result.buffer.get_fn_ptr("add_fn").is_some());
        assert!(result.buffer.get_fn_ptr("mul_fn").is_some());
        assert!(result.buffer.get_fn_ptr("sub_fn").is_some());

        // Verify symbols have distinct offsets (functions are laid out sequentially).
        let symbols: std::collections::HashMap<&str, u64> = result.buffer.symbols().collect();
        assert_eq!(symbols.len(), 3);
    }

    /// Verify that an empty tMIR module returns EmptyModule error.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_compile_module_to_jit_empty_module() {
        use std::collections::HashMap;
        use tmir_build::ModuleBuilder;

        let mb = ModuleBuilder::new("jit_empty");
        let module = mb.build();

        let compiler = Compiler::default_o2();
        let err = compiler
            .compile_module_to_jit(&module, &HashMap::new())
            .unwrap_err();
        assert!(
            matches!(err, CompileError::EmptyModule),
            "expected EmptyModule, got: {err}"
        );
    }

    /// Verify that trace entries are populated when trace_level is Full.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_compile_module_to_jit_with_trace() {
        use std::collections::HashMap;
        use tmir::Ty;
        use tmir_build::ModuleBuilder;

        let mut mb = ModuleBuilder::new("jit_trace");
        {
            let ty = mb.add_func_type(vec![Ty::I64, Ty::I64], vec![Ty::I64]);
            let mut fb = mb.function("add_fn", ty);
            let entry = fb.create_block();
            let a = fb.add_block_param(entry, Ty::I64);
            let b = fb.add_block_param(entry, Ty::I64);
            fb.switch_to_block(entry);
            let result = fb.add(Ty::I64, a, b);
            fb.ret(vec![result]);
            fb.build();
        }
        let module = mb.build();

        let compiler = Compiler::new(CompilerConfig {
            trace_level: CompilerTraceLevel::Full,
            ..CompilerConfig::default()
        });
        let result = compiler
            .compile_module_to_jit(&module, &HashMap::new())
            .unwrap();

        assert!(
            result.trace.is_some(),
            "trace should be present with Full level"
        );
        let trace = result.trace.unwrap();
        assert!(!trace.entries.is_empty(), "trace should have entries");
        // Should have at least an adapter phase and a compile_raw phase.
        let phase_names: Vec<&str> = trace.entries.iter().map(|e| e.phase.as_str()).collect();
        assert!(
            phase_names.contains(&"adapter"),
            "trace should contain adapter phase, got: {:?}",
            phase_names
        );
        assert!(
            phase_names.contains(&"compile_raw"),
            "trace should contain compile_raw phase, got: {:?}",
            phase_names
        );
    }

    /// Verify that proof certificates are populated when emit_proofs is true.
    /// Runs on a thread with enlarged stack because the verifier's recursive
    /// SMT evaluation can overflow the default 8 MiB test thread stack.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_compile_module_to_jit_with_proofs() {
        use std::collections::HashMap;
        use tmir::Ty;
        use tmir_build::ModuleBuilder;

        // 16 MiB stack — verification's recursive evaluator needs headroom.
        let child = std::thread::Builder::new()
            .stack_size(16 * 1024 * 1024)
            .spawn(|| {
                let mut mb = ModuleBuilder::new("jit_proofs");
                {
                    let ty = mb.add_func_type(vec![Ty::I64, Ty::I64], vec![Ty::I64]);
                    let mut fb = mb.function("add_fn", ty);
                    let entry = fb.create_block();
                    let a = fb.add_block_param(entry, Ty::I64);
                    let b = fb.add_block_param(entry, Ty::I64);
                    fb.switch_to_block(entry);
                    let result = fb.add(Ty::I64, a, b);
                    fb.ret(vec![result]);
                    fb.build();
                }
                let module = mb.build();

                let compiler = Compiler::new(CompilerConfig {
                    emit_proofs: true,
                    ..CompilerConfig::default()
                });
                let result = compiler
                    .compile_module_to_jit(&module, &HashMap::new())
                    .unwrap();

                assert!(
                    result.proofs.is_some(),
                    "proofs should be Some when emit_proofs is true"
                );
                let proofs = result.proofs.unwrap();
                assert!(!proofs.is_empty(), "proof certificates should be non-empty");
                for cert in &proofs {
                    assert!(
                        cert.verified,
                        "certificate '{}' should be verified",
                        cert.rule_name
                    );
                }
            })
            .expect("failed to spawn thread with larger stack");
        child.join().expect("test thread panicked");
    }
}
