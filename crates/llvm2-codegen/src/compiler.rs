// llvm2-codegen/compiler.rs - Public compilation API
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

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

use thiserror::Error;

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

    /// Pipeline compilation failed (ISel, regalloc, encoding, etc.).
    #[error("pipeline error: {0}")]
    Pipeline(#[from] PipelineError),

    /// Module contains no functions.
    #[error("empty module: no functions to compile")]
    EmptyModule,
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
}

impl Default for CompilerConfig {
    fn default() -> Self {
        Self {
            opt_level: OptLevel::O2,
            target: Target::Aarch64,
            emit_proofs: false,
            trace_level: CompilerTraceLevel::None,
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
fn generate_proof_certificates(
    func: &llvm2_ir::MachFunction,
) -> Vec<ProofCertificate> {
    use llvm2_verify::function_verifier::{verify_function, InstructionVerificationResult};

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
            InstructionVerificationResult::Failed {
                proof_name,
                detail,
            } => {
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
    pub fn compile(
        &self,
        module: &tmir_func::Module,
    ) -> Result<CompilationResult, CompileError> {
        let total_start = Instant::now();
        let tracing = self.config.trace_level != CompilerTraceLevel::None;
        let mut trace_entries = Vec::new();

        // Phase 1: Translate tMIR module to internal LIR functions.
        let adapter_start = Instant::now();
        let lir_functions = llvm2_lower::translate_module(module)?;
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
        // regalloc, frame lowering, and branch resolution. All functions
        // are then combined into a single Mach-O .o via compile_module()
        // so cross-function BL instructions get proper BRANCH26 relocations.
        let mut prepared_funcs = Vec::with_capacity(lir_functions.len());

        for (lir_func, _proof_ctx) in &lir_functions {
            let func_start = Instant::now();

            let ir_func = pipeline.prepare_function(lir_func)?;

            if tracing {
                trace_entries.push(TraceEntry {
                    phase: "prepare_function".to_string(),
                    duration: func_start.elapsed(),
                    detail: Some(ir_func.name.clone()),
                });
            }

            prepared_funcs.push(ir_func);
        }

        let function_count = prepared_funcs.len();

        // Phase 8-9: Encode all functions and emit a single Mach-O .o file
        // with proper cross-function BRANCH26 relocations.
        let module_start = Instant::now();
        let obj_bytes = pipeline.compile_module(&prepared_funcs)?;

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
            use llvm2_opt::pipeline::{
                OptLevel as OptOptLevel, OptimizationPipeline,
            };
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
        let proofs = if self.config.emit_proofs {
            let mut all_certs = Vec::new();
            for func in &prepared_funcs {
                all_certs.extend(generate_proof_certificates(func));
            }
            Some(all_certs)
        } else {
            None
        };

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
            use llvm2_opt::pipeline::{
                OptLevel as OptOptLevel, OptimizationPipeline,
            };
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

        let proofs = if self.config.emit_proofs {
            Some(generate_proof_certificates(ir_func))
        } else {
            None
        };

        Ok(CompilationResult {
            object_code: obj_bytes,
            metrics,
            trace,
            proofs,
        })
    }

    /// Build the internal [`Pipeline`] from the compiler configuration.
    fn build_pipeline(&self) -> Pipeline {
        Pipeline::new(PipelineConfig {
            opt_level: self.config.opt_level,
            emit_debug: false,
            verify_dispatch: crate::pipeline::DispatchVerifyMode::FallbackOnFailure,
            verify: false,
            enable_post_ra_opt: self.config.opt_level != crate::pipeline::OptLevel::O0,
            use_pressure_aware_scheduler: matches!(
                self.config.opt_level,
                crate::pipeline::OptLevel::O2 | crate::pipeline::OptLevel::O3
            ),
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = CompilerConfig::default();
        assert_eq!(config.opt_level, OptLevel::O2);
        assert_eq!(config.target, Target::Aarch64);
        assert!(!config.emit_proofs);
        assert_eq!(config.trace_level, CompilerTraceLevel::None);
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
        };
        let compiler = Compiler::new(config);
        assert_eq!(compiler.config().opt_level, OptLevel::O0);
        assert_eq!(compiler.config().target, Target::X86_64);
        assert!(compiler.config().emit_proofs);
        assert_eq!(compiler.config().trace_level, CompilerTraceLevel::Full);
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

        assert!(!result.object_code.is_empty(), "should produce Mach-O bytes");
        assert!(result.metrics.code_size_bytes > 0);
        assert_eq!(result.metrics.function_count, 1);
        assert!(result.trace.is_none(), "trace should be None with default config");
        assert!(result.proofs.is_none(), "proofs should be None with default config");
    }

    #[test]
    fn test_compile_ir_function_with_trace() {
        let mut ir_func = crate::pipeline::build_add_test_function();
        let compiler = Compiler::new(CompilerConfig {
            trace_level: CompilerTraceLevel::Summary,
            ..CompilerConfig::default()
        });
        let result = compiler.compile_ir_function(&mut ir_func).unwrap();

        assert!(result.trace.is_some(), "trace should be present with Summary level");
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

        assert!(result.proofs.is_some(), "proofs field should be Some when emit_proofs is true");
        let proofs = result.proofs.unwrap();
        // The add test function contains an AddRR instruction which has a
        // verified proof obligation in the proof database.
        assert!(!proofs.is_empty(), "proof certificates should be non-empty for add function");
        // Every certificate that was emitted should be verified (no failures).
        for cert in &proofs {
            assert!(cert.verified, "certificate '{}' should be verified", cert.rule_name);
            assert!(!cert.rule_name.is_empty(), "rule_name should not be empty");
            assert!(!cert.category.is_empty(), "category should not be empty");
            assert!(!cert.strength.is_empty(), "strength should not be empty");
            assert!(!cert.function_name.is_empty(), "function_name should not be empty");
        }
    }

    #[test]
    fn test_compiler_trace_level_equality() {
        assert_eq!(CompilerTraceLevel::None, CompilerTraceLevel::None);
        assert_ne!(CompilerTraceLevel::None, CompilerTraceLevel::Summary);
        assert_ne!(CompilerTraceLevel::Summary, CompilerTraceLevel::Full);
    }
}
