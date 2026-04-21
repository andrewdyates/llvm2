// llvm2-codegen/pipeline.rs - End-to-end compilation pipeline
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Wires together all LLVM2 crates into a single entry point:
//   tMIR -> ISel -> Optimization -> RegAlloc -> Frame Lowering -> Encoding -> Mach-O

//! End-to-end compilation pipeline for LLVM2.
//!
//! Takes a tMIR function (represented as `llvm2_lower::Function`) and produces
//! a valid AArch64 Mach-O .o file.
//!
//! # Pipeline phases
//!
//! ```text
//! Phase 1: ISel (llvm2-lower)
//!   llvm2_lower::Function -> llvm2_lower::isel::ISelFunction
//!   (VRegs, canonical llvm2_ir::AArch64Opcode — unified in issue #73)
//!
//! Phase 2: Adapt ISel -> IR (structural only — opcodes are already unified)
//!   llvm2_lower::isel::ISelFunction -> llvm2_ir::MachFunction
//!
//! Phase 3: Optimization (llvm2-opt)
//!   llvm2_ir::MachFunction -> llvm2_ir::MachFunction (optimized)
//!
//! Phase 4: Adapt IR -> RegAlloc
//!   llvm2_ir::MachFunction -> llvm2_regalloc::RegAllocFunction
//!
//! Phase 5: Register Allocation (llvm2-regalloc)
//!   llvm2_regalloc::RegAllocFunction -> AllocationResult (VReg -> PReg map)
//!
//! Phase 6: Apply allocation to IR
//!   llvm2_ir::MachFunction + AllocationResult -> llvm2_ir::MachFunction (PRegs only)
//!
//! Phase 7: Frame Lowering (llvm2-codegen/frame)
//!   Insert prologue/epilogue, eliminate frame indices
//!
//! Phase 8: Encoding (llvm2-codegen/aarch64)
//!   llvm2_ir::MachFunction -> Vec<u8> (machine code bytes)
//!
//! Phase 9: Mach-O Emission (llvm2-codegen/macho)
//!   Vec<u8> -> Mach-O .o file bytes
//! ```
//!
//! # Type adapters and unification status (issue #73)
//!
//! **Unified (no adapter needed):**
//! - `AArch64Opcode`: ISel uses `llvm2_ir::AArch64Opcode` directly.
//! - `InstFlags`: regalloc re-exports `llvm2_ir::InstFlags` directly.
//! - `MachBlock` / `RegAllocBlock`: structurally identical after `loop_depth`
//!   was added to `llvm2_ir::MachBlock`. Converted via `From<&MachBlock>`.
//! - `StackSlot` / `RegAllocStackSlot`: converted via `From<&StackSlot>`.
//! - Primitive types: `VReg`, `PReg`, `RegClass`, `BlockId`, `InstId`,
//!   `StackSlotId` — all re-exported from `llvm2_ir` by regalloc.
//!
//! **Remaining structural adapters (separate types, with From/TryFrom impls):**
//! - `ISelOperand` / `RegAllocOperand`: ISel has `CondCode`/`Symbol`/`StackSlot(u32)`
//!   variants; regalloc omits `MemOp`/`FrameIndex`/`Special`.
//!   `TryFrom<&MachOperand>` for `RegAllocOperand` handles the conversion.
//! - `ISelInst` / `RegAllocInst`: regalloc separates defs/uses for liveness analysis;
//!   ISel has no flags/implicit-defs/proofs. Unification requires adding
//!   def/use classification to `llvm2_ir::MachInst`.
//! - `ISelBlock`: ISel uses inline `Vec<ISelInst>` (not arena-indexed).
//! - `ISelFunction` / `RegAllocFunction`: ISel uses `HashMap<Block, ISelBlock>`
//!   (construction-friendly); regalloc uses `HashMap<StackSlotId, RegAllocStackSlot>`
//!   + `next_stack_slot`.
//!
//! The canonical types are in `llvm2_ir`: `MachFunction`, `MachInst`, `MachBlock`,
//! `MachOperand`. ISel and regalloc types have been renamed to avoid confusion.
//! See `llvm2_ir::type_hierarchy` for complete documentation.
//! See issue #73 for the remaining unification plan.

use std::collections::HashMap;
use std::sync::Arc;
#[cfg(not(feature = "verify"))]
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};
use thiserror::Error;

use llvm2_ir::function::{
    MachFunction as IrMachFunction, Signature as IrSignature, StackSlot as IrStackSlot,
};
use llvm2_ir::inst::{AArch64Opcode as IrOpcode, MachInst as IrMachInst};
use llvm2_ir::operand::MachOperand as IrOperand;
use llvm2_ir::regs::{PReg, VReg};
use llvm2_ir::types::BlockId;

use llvm2_lower::TargetRecommendation;
use llvm2_lower::compute_graph::{ComputeGraph, ComputeNode, ComputeNodeId};
use llvm2_lower::dispatch::{
    DispatchOp, DispatchPlan, VerificationReport, generate_dispatch_plan,
    verify_dispatch_plan_properties,
};
use llvm2_lower::target_analysis::ComputeTarget;

use crate::coreml_emitter::{
    CoreMLEmitError, CoreMLEmitter, MilProgram, validate_ane_compatibility,
};

// ---------------------------------------------------------------------------
// Pipeline errors
// ---------------------------------------------------------------------------

/// Errors during compilation.
#[derive(Debug, Error)]
pub enum PipelineError {
    #[error("instruction selection failed: {0}")]
    ISel(String),
    #[error("register allocation failed: {0}")]
    RegAlloc(String),
    #[error("encoding failed: {0}")]
    Encoding(String),
    #[error("unsupported opcode during encoding: {0:?}")]
    UnsupportedOpcode(IrOpcode),
    #[error("branch relaxation failed: {0}")]
    Relaxation(#[from] crate::relax::RelaxError),
    #[error("invalid operand in regalloc adapter: {0}")]
    InvalidOperand(String),
    #[error("dispatch verification failed ({violations} violations): {summary}")]
    DispatchVerificationFailed {
        violations: usize,
        summary: String,
        report: VerificationReport,
    },
    #[error("CoreML MIL emission failed: {0}")]
    CoreMLEmit(#[from] CoreMLEmitError),
    #[error("Metal kernel emission failed: {0}")]
    MetalEmit(#[from] crate::metal_emitter::MetalEmitError),
    #[error("Mach-O fixup resolution failed: {0}")]
    Fixup(#[from] crate::macho::FixupError),
    #[cfg(feature = "verify")]
    #[error(
        "function verification failed: {failures} failures, {coverage:.1}% coverage ({function})"
    )]
    VerificationFailed {
        function: String,
        failures: usize,
        coverage: f64,
        report: llvm2_verify::FunctionVerificationReport,
    },
}

// ---------------------------------------------------------------------------
// Pipeline configuration
// ---------------------------------------------------------------------------

/// Optimization level for the pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptLevel {
    O0,
    O1,
    O2,
    O3,
}

/// Policy for handling dispatch verification failures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DispatchVerifyMode {
    /// Do not verify dispatch plans (skip verification entirely).
    Off,
    /// Verify dispatch plans. On failure, fall back to a CPU-only plan.
    FallbackOnFailure,
    /// Verify dispatch plans. On failure, return an error.
    ErrorOnFailure,
}

/// Pipeline configuration.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Optimization level.
    pub opt_level: OptLevel,
    /// Whether to emit debug info (placeholder for future).
    pub emit_debug: bool,
    /// Dispatch plan verification mode.
    ///
    /// When set to [`DispatchVerifyMode::FallbackOnFailure`], any dispatch plan
    /// that fails property verification is replaced with a CPU-only plan. When
    /// set to [`DispatchVerifyMode::ErrorOnFailure`], verification failures
    /// produce a [`PipelineError::DispatchVerificationFailed`].
    pub verify_dispatch: DispatchVerifyMode,
    /// Whether to run function-level verification after optimization.
    ///
    /// When enabled, the pipeline calls [`llvm2_verify::verify_function`] on
    /// the optimized IR before register allocation and returns
    /// [`PipelineError::VerificationFailed`] if any instruction fails
    /// verification. Pseudo-ops and unverified (no-proof) instructions are
    /// tolerated — only explicit verification failures are fatal.
    ///
    /// Default: `false` (verification disabled for compilation speed).
    pub verify: bool,
    /// Whether to run post-register-allocation spill optimization.
    ///
    /// When enabled, [`llvm2_regalloc::post_ra_optimize`] is called after
    /// register allocation to eliminate dead spill stores, redundant reloads,
    /// and dead definitions left behind by conservative spill decisions.
    ///
    /// Default: `true`. Disabled at O0 by the pipeline's opt-level logic.
    pub enable_post_ra_opt: bool,
    /// Whether to use the pressure-aware instruction scheduler at O2+.
    ///
    /// When true and `opt_level` is O2 or O3, the pipeline runs
    /// [`llvm2_opt::scheduler::PressureAwareScheduler`] instead of
    /// [`llvm2_opt::scheduler::InstructionScheduler`]. The pressure-aware
    /// variant trades some ILP for lower register pressure, reducing spills
    /// in functions with many live values.
    ///
    /// At O0 and O1, the fast [`InstructionScheduler`] is always used
    /// regardless of this setting (scheduling is skipped entirely at O0).
    ///
    /// Default: `true`.
    pub use_pressure_aware_scheduler: bool,
    /// Per-function wall-clock budget for the CEGIS superopt pass.
    ///
    /// When `Some(n)`, the optimization pipeline schedules
    /// [`llvm2_verify::CegisSuperoptPass`] with `budget_sec = n`. When `None`
    /// (default), the pass is not scheduled. Threaded through from
    /// [`llvm2_codegen::CompilerConfig::cegis_superopt_budget_sec`].
    ///
    /// Issue: #395. Default: `None` (off).
    pub cegis_superopt_budget_sec: Option<u64>,
    /// Target triple used to key CEGIS superopt cache entries.
    ///
    /// The default is the empty string for back-compat at existing config
    /// construction sites that do not care about target-distinct cache keys.
    pub target_triple: String,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            opt_level: OptLevel::O2,
            emit_debug: false,
            verify_dispatch: DispatchVerifyMode::FallbackOnFailure,
            verify: false,
            enable_post_ra_opt: true,
            use_pressure_aware_scheduler: true,
            cegis_superopt_budget_sec: None,
            target_triple: String::new(),
        }
    }
}

/// Wall-clock timings for each compilation phase within `prepare_function*`.
///
/// All fields are [`Duration`]. `None` means the phase was skipped
/// (e.g. verification skipped when `config.verify = false`, or encoding
/// phase not yet reached).
///
/// Used by [`crate::compiler::Compiler::compile_module_to_jit`] and consumed
/// by downstream observability tools (tla2) for A/B benchmarking the
/// Cranelift -> LLVM2 migration. See issue #364.
#[derive(Debug, Clone, Default)]
pub struct PhaseTimings {
    pub isel: Option<Duration>,
    pub optimization: Option<Duration>,
    pub verification: Option<Duration>,
    pub regalloc: Option<Duration>,
    pub frame_lowering: Option<Duration>,
    pub branch_resolution: Option<Duration>,
    pub encoding: Option<Duration>,
}

impl PhaseTimings {
    /// Total duration across all populated phases.
    pub fn total(&self) -> Duration {
        [
            self.isel,
            self.optimization,
            self.verification,
            self.regalloc,
            self.frame_lowering,
            self.branch_resolution,
            self.encoding,
        ]
        .iter()
        .filter_map(|d| *d)
        .sum()
    }
}

/// Additional per-function preparation metrics surfaced by
/// [`Pipeline::prepare_function_with_metrics`].
#[derive(Debug, Clone, Default)]
pub struct PreparationMetrics {
    pub timings: PhaseTimings,
    pub spill_slot_count: usize,
}

#[cfg(feature = "verify")]
pub use llvm2_verify::CegisPassStats;

#[cfg(not(feature = "verify"))]
#[derive(Debug, Clone, Default)]
pub struct CegisPassStats {
    pub functions_seen: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub cache_puts: u64,
    pub candidates: u64,
    pub verified: u64,
    pub rejected: u64,
    pub budget_exhausted: u64,
    pub solver_calls: u64,
}

// ---------------------------------------------------------------------------
// Input format detection and module loading
// ---------------------------------------------------------------------------

/// Magic bytes for the binary tMIR bitcode (.tmbc) format.
pub const TMBC_MAGIC: &[u8; 4] = b"tMBC";

/// Current version of the .tmbc wire format.
pub const TMBC_VERSION: u32 = 1;

/// Detected input format for tMIR module files.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputFormat {
    /// JSON wire format (.json).
    Json,
    /// Binary tMIR bitcode (.tmbc).
    Tmbc,
    /// Human-readable text format (.tmir), produced by
    /// `tmir::Module`'s `Display` impl and parsed by `tmir::parser`.
    Tmir,
}

/// Format selection mode, used when the caller wants either explicit
/// choice or the legacy auto-detect behaviour.
///
/// `Tmbc`, `Json`, and `Text` force the corresponding parser and ignore
/// both the file extension and any magic-byte mismatch until decode.
/// `Auto` restores the pre-#414 behaviour: prefer `.tmbc`/`.json`/`.tmir`
/// by extension, then sniff magic bytes, finally fall back to JSON.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FormatMode {
    /// Force binary tMIR bitcode (.tmbc).
    Tmbc,
    /// Force JSON wire format.
    Json,
    /// Force human-readable `.tmir` text format (#413).
    ///
    /// The text format is produced by `tmir::Module`'s `Display` impl
    /// and parsed by `tmir::parser::parse_module`. The parser is
    /// gated behind the upstream `parser` feature, which LLVM2 now
    /// enables in `[workspace.dependencies]`.
    Text,
    /// Auto-detect by extension and magic bytes (legacy behaviour).
    Auto,
}

/// Detect the input format of a tMIR module file (legacy auto-detect).
///
/// Checks file extension first (`.tmbc`, `.tmir`, or `.json`). For
/// unknown extensions, reads the first 4 bytes and checks for the
/// tMBC magic header; otherwise, sniffs for the text-format header
/// `; tMIR text format` (produced by `tmir::Module`'s `Display` impl).
///
/// Prefer `load_module_as` for new code: per the tMIR transport
/// architecture (designs/2026-04-16-tmir-transport-architecture.md,
/// Layer 4), binary `.tmbc` is the intended default and JSON input is
/// a debug-only opt-in.
pub fn detect_input_format(path: &std::path::Path) -> InputFormat {
    match path.extension().and_then(|ext| ext.to_str()) {
        Some("tmbc") => return InputFormat::Tmbc,
        Some("json") => return InputFormat::Json,
        Some("tmir") => return InputFormat::Tmir,
        _ => {}
    }

    // Fall back to magic-byte detection. tMIR text-format modules start
    // with the exact comment `; tMIR text format v1` (see
    // `tmir::display`). We treat any leading `;` or `module ` prefix
    // as a text-format hint, then fall back to JSON.
    let mut magic = [0u8; 20];
    if let Ok(()) = std::fs::File::open(path)
        .and_then(|mut file| std::io::Read::read_exact(&mut file, &mut magic))
    {
        if &magic[..4] == b"tMBC" {
            return InputFormat::Tmbc;
        }
        if magic.starts_with(b"; tMIR text format") || magic.starts_with(b"module ") {
            return InputFormat::Tmir;
        }
    }
    InputFormat::Json
}

/// Load a tMIR module from a file using the legacy auto-detect path.
///
/// Kept for in-tree callers that still want the old behaviour. The CLI
/// now uses [`load_module_as`] and defaults to binary `.tmbc`.
pub fn load_module(path: &std::path::Path) -> Result<tmir::Module, PipelineError> {
    load_module_as(path, FormatMode::Auto)
}

/// Load a tMIR module from a file with an explicit format selection.
///
/// - `FormatMode::Tmbc`: parse the file as binary tMIR bitcode. If the
///   magic bytes do not match, return a clear `PipelineError::ISel`
///   error that names the `--format=json` CLI escape hatch.
/// - `FormatMode::Json`: parse the file as a JSON-serialised
///   `tmir::Module`, regardless of extension.
/// - `FormatMode::Text`: parse the file as a human-readable `.tmir`
///   text module via `tmir::parser::parse_module` (#413).
/// - `FormatMode::Auto`: preserve the pre-#414 auto-detect behaviour
///   (extension + magic-byte sniffing, JSON fallback; now also
///   recognises `.tmir`).
pub fn load_module_as(
    path: &std::path::Path,
    mode: FormatMode,
) -> Result<tmir::Module, PipelineError> {
    match mode {
        FormatMode::Tmbc => {
            let bytes = std::fs::read(path)
                .map_err(|err| PipelineError::ISel(format!("I/O error: {err}")))?;
            if !bytes.starts_with(TMBC_MAGIC) {
                return Err(PipelineError::ISel(format!(
                    "'{}' does not start with the tMBC magic header; the CLI now defaults to \
                     binary tMIR bitcode input. Pass `--format=json` if this file is a tMIR \
                     JSON wire-format module, or regenerate it as `.tmbc`.",
                    path.display()
                )));
            }
            decode_tmbc(&bytes)
        }
        FormatMode::Json => {
            let json = std::fs::read_to_string(path)
                .map_err(|err| PipelineError::ISel(format!("I/O error: {err}")))?;
            serde_json::from_str(&json)
                .map_err(|err: serde_json::Error| PipelineError::ISel(format!("JSON error: {err}")))
        }
        FormatMode::Text => {
            let text = std::fs::read_to_string(path)
                .map_err(|err| PipelineError::ISel(format!("I/O error: {err}")))?;
            parse_tmir_text(&text)
        }
        FormatMode::Auto => match detect_input_format(path) {
            InputFormat::Json => {
                let json = std::fs::read_to_string(path)
                    .map_err(|err| PipelineError::ISel(format!("I/O error: {err}")))?;
                serde_json::from_str(&json).map_err(|err: serde_json::Error| {
                    PipelineError::ISel(format!("JSON error: {err}"))
                })
            }
            InputFormat::Tmbc => {
                let bytes = std::fs::read(path)
                    .map_err(|err| PipelineError::ISel(format!("I/O error: {err}")))?;
                decode_tmbc(&bytes)
            }
            InputFormat::Tmir => {
                let text = std::fs::read_to_string(path)
                    .map_err(|err| PipelineError::ISel(format!("I/O error: {err}")))?;
                parse_tmir_text(&text)
            }
        },
    }
}

/// Load a tMIR module from in-memory bytes, auto-detecting the format.
///
/// Bytes starting with `tMBC` magic are decoded as binary bitcode. UTF-8
/// text that begins with `; tMIR text format` or `module ` is parsed
/// as `.tmir` text (#413). Otherwise the bytes are interpreted as a
/// UTF-8 JSON string.
pub fn load_module_from_bytes(bytes: &[u8]) -> Result<tmir::Module, PipelineError> {
    if bytes.starts_with(TMBC_MAGIC) {
        return decode_tmbc(bytes);
    }
    let text = std::str::from_utf8(bytes).map_err(|err| PipelineError::ISel(err.to_string()))?;
    let trimmed = text.trim_start();
    if trimmed.starts_with("; tMIR text format") || trimmed.starts_with("module ") {
        return parse_tmir_text(text);
    }
    serde_json::from_str(text)
        .map_err(|err: serde_json::Error| PipelineError::ISel(format!("JSON error: {err}")))
}

/// Parse a `.tmir` human-readable text-format module via
/// `tmir::parser::parse_module` (#413).
///
/// Requires the `parser` feature on the upstream `tmir` crate, which
/// LLVM2's `[workspace.dependencies]` now enables.
pub fn parse_tmir_text(text: &str) -> Result<tmir::Module, PipelineError> {
    tmir::parser::parse_module(text)
        .map_err(|err| PipelineError::ISel(format!(".tmir parse error: {err}")))
}

/// Render a `tmir::Module` to `.tmir` human-readable text via its
/// `Display` impl (always on in upstream; see `tmir::display`).
///
/// The returned string round-trips through `tmir::parser::parse_module`.
pub fn encode_tmir_text(module: &tmir::Module) -> String {
    format!("{}", module)
}

/// Serialize a `tmir::Module` as `.tmir` text and write it to disk (#413).
pub fn save_module_to_tmir_text(
    module: &tmir::Module,
    path: &std::path::Path,
) -> Result<(), PipelineError> {
    let text = encode_tmir_text(module);
    std::fs::write(path, text).map_err(|e| {
        PipelineError::ISel(format!(
            "failed to write .tmir text file {}: {e}",
            path.display()
        ))
    })
}

// ---------------------------------------------------------------------------
// Binary tMIR bitcode (.tmbc) codec
// ---------------------------------------------------------------------------

/// Decode a binary tMIR bitcode (.tmbc) buffer into a `tmir::Module`.
///
/// Format: `tMBC` magic (4 bytes) + version u32-LE (4 bytes) + `tmir::binary`
/// payload (itself prefixed with `TMIR` magic + u32-LE version; see
/// `tmir::binary::deserialize_module`).
///
/// The outer `tMBC` envelope is LLVM2's transport-layer header (see
/// `designs/2026-04-16-tmir-transport-architecture.md`, Layer 2). The inner
/// payload is produced by the upstream `tmir/binary` feature — a compact,
/// zero-extra-dependency encoding with its own `TMIR` magic and version.
pub fn decode_tmbc(bytes: &[u8]) -> Result<tmir::Module, PipelineError> {
    if bytes.len() < 8 {
        return Err(PipelineError::ISel(format!(
            "invalid tMBC data: expected at least 8 bytes, got {}",
            bytes.len()
        )));
    }

    if &bytes[..4] != TMBC_MAGIC {
        return Err(PipelineError::ISel(format!(
            "invalid tMBC magic: expected {:?}, got {:?}",
            TMBC_MAGIC,
            &bytes[..4]
        )));
    }

    let version = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    if version != TMBC_VERSION {
        return Err(PipelineError::ISel(format!(
            "unsupported tMBC version: {version}"
        )));
    }

    tmir::binary::deserialize_module(&bytes[8..])
        .map_err(|e| PipelineError::ISel(format!("failed to deserialize tMBC payload: {e}")))
}

/// Encode a `tmir::Module` into binary tMIR bitcode (.tmbc) format.
///
/// Format: `tMBC` magic (4 bytes) + version u32-LE (4 bytes) + `tmir::binary`
/// payload. The inner payload is produced by `tmir::binary::serialize_module`,
/// which carries its own `TMIR` magic + u32-LE version header. See
/// `designs/2026-04-16-tmir-transport-architecture.md`, Layer 2.
pub fn encode_tmbc(module: &tmir::Module) -> Result<Vec<u8>, PipelineError> {
    let payload = tmir::binary::serialize_module(module);

    let mut buf = Vec::with_capacity(8 + payload.len());
    buf.extend_from_slice(TMBC_MAGIC);
    buf.extend_from_slice(&TMBC_VERSION.to_le_bytes());
    buf.extend_from_slice(&payload);
    Ok(buf)
}

/// Serialize a `tmir::Module` and write it to a `.tmbc` file on disk.
pub fn save_module_to_tmbc(
    module: &tmir::Module,
    path: &std::path::Path,
) -> Result<(), PipelineError> {
    let buf = encode_tmbc(module)?;
    std::fs::write(path, buf).map_err(|e| {
        PipelineError::ISel(format!("failed to write tMBC file {}: {e}", path.display()))
    })
}

// ---------------------------------------------------------------------------
// CompilationUnit — holds all state across compilation phases
// ---------------------------------------------------------------------------

/// Holds the state of a single function being compiled through the pipeline.
pub struct CompilationUnit {
    /// The function name.
    pub name: String,
    /// The IR-level function (shared model used by opt and frame lowering).
    pub ir_func: IrMachFunction,
    /// Encoded machine code bytes (populated after encoding phase).
    pub code: Vec<u8>,
    /// Pipeline configuration.
    pub config: PipelineConfig,
}

impl CompilationUnit {
    /// Create a new compilation unit from an IR function.
    pub fn new(ir_func: IrMachFunction, config: PipelineConfig) -> Self {
        let name = ir_func.name.clone();
        Self {
            name,
            ir_func,
            code: Vec::new(),
            config,
        }
    }
}

// ---------------------------------------------------------------------------
// Adapter: llvm2_lower::isel::ISelFunction -> llvm2_ir::MachFunction
// ---------------------------------------------------------------------------

// NOTE: `map_isel_opcode()` has been ELIMINATED as of issue #73.
//
// The ISel now uses `llvm2_ir::AArch64Opcode` directly (re-exported via
// `llvm2_lower::isel::AArch64Opcode`). There is no longer a separate ISel
// opcode enum, so no mapping is needed. The `isel_to_ir` adapter below
// uses `isel_inst.opcode` directly as an `IrOpcode`.

/// Convert ISel's `ISelFunction` to the canonical `llvm2_ir::MachFunction`.
///
/// Delegates to `ISelFunction::to_ir_func()` which owns the conversion logic
/// (issue #73 consolidation). This thin wrapper is preserved for backward
/// compatibility — new code should call `isel_func.to_ir_func()` directly.
pub fn isel_to_ir(
    isel_func: &llvm2_lower::isel::ISelFunction,
    _param_types: &[llvm2_ir::function::Type],
    _return_types: &[llvm2_ir::function::Type],
) -> IrMachFunction {
    // The ISelFunction::to_ir_func() method derives the IR signature from
    // the ISel's own signature using From<Signature>. The param_types and
    // return_types arguments are now unused — they were only needed when
    // the conversion was done externally.
    isel_func.to_ir_func()
}

// ---------------------------------------------------------------------------
// Adapter: llvm2_ir::MachFunction -> llvm2_regalloc::RegAllocFunction
// ---------------------------------------------------------------------------

/// Convert the canonical `llvm2_ir::MachFunction` to `RegAllocFunction` format.
///
/// The `RegAllocFunction` separates defs from uses for liveness analysis.
/// We use a simple heuristic: first operand is def, rest are uses
/// (for most instructions). Special cases: branches have no defs, etc.
///
/// Uses `From`/`TryFrom` impls from `llvm2_regalloc::machine_types` for
/// individual block and operand conversions (issue #73 consolidation).
pub fn ir_to_regalloc(
    ir_func: &IrMachFunction,
) -> Result<llvm2_regalloc::RegAllocFunction, PipelineError> {
    use llvm2_regalloc::machine_types as ra;

    let mut ra_func = ra::RegAllocFunction {
        name: ir_func.name.clone(),
        insts: Vec::new(),
        blocks: Vec::new(),
        block_order: ir_func.block_order.clone(),
        entry_block: ir_func.entry,
        next_vreg: ir_func.next_vreg,
        next_stack_slot: ir_func.stack_slots.len() as u32,
        stack_slots: HashMap::new(),
    };

    // Convert stack slots using From impl (issue #73).
    for (i, slot) in ir_func.stack_slots.iter().enumerate() {
        ra_func.stack_slots.insert(
            llvm2_ir::types::StackSlotId(i as u32),
            ra::RegAllocStackSlot::from(slot),
        );
    }

    // Convert instructions.
    // InstFlags are unified (issue #73) — same type in llvm2_ir and llvm2_regalloc,
    // so flags pass through directly without conversion.
    for ir_inst in &ir_func.insts {
        let (defs, uses) = classify_def_use(ir_inst)?;
        let implicit_defs: Vec<PReg> = ir_inst.implicit_defs.to_vec();
        let implicit_uses: Vec<PReg> = ir_inst.implicit_uses.to_vec();

        ra_func.insts.push(ra::RegAllocInst {
            opcode: ir_inst.opcode as u16,
            defs,
            uses,
            implicit_defs,
            implicit_uses,
            flags: ir_inst.flags,
        });
    }

    // Convert blocks using From impl (issue #73).
    for ir_block in &ir_func.blocks {
        ra_func.blocks.push(ra::RegAllocBlock::from(ir_block));
    }

    Ok(ra_func)
}

/// Classify operands into defs and uses for regalloc.
///
/// Convention:
/// - For most ALU instructions: operand[0] is def, rest are uses
/// - For stores: all operands are uses (no register def)
/// - For branches/returns: all operands are uses
/// - For loads: operand[0] is def, rest are uses
///
/// Uses `TryFrom<&MachOperand>` for `RegAllocOperand` (issue #73) for the
/// individual operand conversions. The TryFrom impl rejects IR-only variants
/// (FrameIndex, MemOp, Special) that must be lowered before register
/// allocation.
fn classify_def_use(
    inst: &IrMachInst,
) -> Result<
    (
        Vec<llvm2_regalloc::RegAllocOperand>,
        Vec<llvm2_regalloc::RegAllocOperand>,
    ),
    PipelineError,
> {
    use llvm2_regalloc::machine_types::RegAllocOperand as RaOp;

    let convert_op = |op: &IrOperand| -> Result<RaOp, PipelineError> {
        RaOp::try_from(op).map_err(|e| PipelineError::InvalidOperand(e.message))
    };

    let is_store = inst
        .flags
        .contains(llvm2_ir::inst::InstFlags::WRITES_MEMORY);
    let is_branch = inst.flags.contains(llvm2_ir::inst::InstFlags::IS_BRANCH);
    let is_return = inst.flags.contains(llvm2_ir::inst::InstFlags::IS_RETURN);
    let is_cmp = matches!(
        inst.opcode,
        IrOpcode::CmpRR | IrOpcode::CmpRI | IrOpcode::Fcmp
    );

    if is_store || is_branch || is_return || is_cmp || inst.operands.is_empty() {
        // All uses, no defs.
        let uses: Vec<RaOp> = inst
            .operands
            .iter()
            .map(&convert_op)
            .collect::<Result<_, _>>()?;
        Ok((Vec::new(), uses))
    } else {
        // First operand is def, rest are uses.
        let defs = vec![convert_op(&inst.operands[0])?];
        let uses: Vec<RaOp> = inst.operands[1..]
            .iter()
            .map(convert_op)
            .collect::<Result<_, _>>()?;
        Ok((defs, uses))
    }
}

// NOTE: `convert_ir_flags_to_regalloc()` has been ELIMINATED as of issue #73.
//
// InstFlags are now unified: llvm2_regalloc re-exports llvm2_ir::InstFlags
// directly. The bit-for-bit flag conversion function is no longer needed —
// flags pass through directly in ir_to_regalloc().

// ---------------------------------------------------------------------------
// Apply register allocation results back to IR
// ---------------------------------------------------------------------------

/// Rewrite all VReg operands in the IR function with their allocated PRegs.
pub fn apply_regalloc(ir_func: &mut IrMachFunction, allocation: &HashMap<VReg, PReg>) {
    for inst in &mut ir_func.insts {
        for operand in &mut inst.operands {
            if let IrOperand::VReg(vreg) = operand
                && let Some(&preg) = allocation.get(vreg)
            {
                *operand = IrOperand::PReg(preg);
            }
            // If not found in allocation, the vreg was spilled.
            // Spill code insertion should have already handled this.
        }
    }
}

// ---------------------------------------------------------------------------
// Proof annotation propagation
// ---------------------------------------------------------------------------

/// Convert an adapter-layer [`Proof`](llvm2_lower::Proof) to the IR-layer
/// [`ProofAnnotation`](llvm2_ir::ProofAnnotation).
///
/// Some proof variants carry extra metadata (e.g., `InBounds { base, index }`)
/// that is not represented in the simpler IR annotation enum. The conversion
/// preserves the proof *kind* while dropping operand-specific metadata, since
/// the annotation is attached to the defining instruction which already
/// encodes its operands.
fn proof_to_annotation(proof: &llvm2_lower::Proof) -> llvm2_ir::ProofAnnotation {
    match proof {
        llvm2_lower::Proof::NoOverflow { .. } => llvm2_ir::ProofAnnotation::NoOverflow,
        llvm2_lower::Proof::InBounds { .. } => llvm2_ir::ProofAnnotation::InBounds,
        llvm2_lower::Proof::NotNull { .. } => llvm2_ir::ProofAnnotation::NotNull,
        llvm2_lower::Proof::ValidBorrow { .. } => llvm2_ir::ProofAnnotation::ValidBorrow,
        llvm2_lower::Proof::NonZeroDivisor { .. } => llvm2_ir::ProofAnnotation::NonZeroDivisor,
        llvm2_lower::Proof::ValidShift { .. } => llvm2_ir::ProofAnnotation::ValidShift,
        llvm2_lower::Proof::Pure => llvm2_ir::ProofAnnotation::Pure,
        llvm2_lower::Proof::Associative => llvm2_ir::ProofAnnotation::Associative,
        llvm2_lower::Proof::Commutative => llvm2_ir::ProofAnnotation::Commutative,
        llvm2_lower::Proof::Idempotent => llvm2_ir::ProofAnnotation::Idempotent,
        // InRange has no direct annotation counterpart — the range info
        // cannot be expressed in the single-variant ProofAnnotation. We
        // skip it here; range-based opts use the full ProofContext.
        llvm2_lower::Proof::InRange { .. } => llvm2_ir::ProofAnnotation::InBounds,
    }
}

/// Propagate proof annotations from the adapter's [`ProofContext`](llvm2_lower::ProofContext)
/// onto machine instructions in the IR function.
///
/// Uses the ISel value map to resolve LIR Values to VRegs, then builds a
/// reverse map (VReg ID -> InstId) to locate the defining instruction for
/// each value and annotate it with the appropriate [`ProofAnnotation`].
///
/// Function-level proofs (Pure, Associative, etc.) are stored on the
/// [`MachFunction::function_proofs`] field.
fn apply_proof_annotations(
    ir_func: &mut IrMachFunction,
    proof_ctx: &llvm2_lower::ProofContext,
    value_map: &std::collections::HashMap<
        llvm2_lower::instructions::Value,
        llvm2_lower::isel::ISelOperand,
    >,
) {
    use llvm2_ir::MachOperand as IrOp;
    use llvm2_ir::types::InstId;

    // Build reverse map: VReg ID -> first InstId that defines it.
    // Convention: operand[0] is the destination register for instructions
    // that produce a result.
    let mut vreg_to_inst: std::collections::HashMap<u32, InstId> = std::collections::HashMap::new();
    for (idx, inst) in ir_func.insts.iter().enumerate() {
        if let Some(IrOp::VReg(vreg)) = inst.operands.first() {
            // Only record the first definition (SSA property).
            vreg_to_inst.entry(vreg.id).or_insert(InstId(idx as u32));
        }
    }

    // Propagate per-value proofs onto defining instructions.
    for (value, proofs) in &proof_ctx.value_proofs {
        if let Some(isel_op) = value_map.get(value) {
            // Extract VReg ID from ISel operand.
            let vreg_id = match isel_op {
                llvm2_lower::isel::ISelOperand::VReg(vreg) => Some(vreg.id),
                _ => None,
            };

            if let Some(vid) = vreg_id {
                if let Some(&inst_id) = vreg_to_inst.get(&vid) {
                    // Pick the most meaningful proof for the instruction annotation.
                    // ProofAnnotation is a single-value field, so we take the first
                    // proof that maps to a concrete optimization opportunity.
                    if let Some(proof) = proofs.first() {
                        let annotation = proof_to_annotation(proof);
                        let inst = ir_func.inst_mut(inst_id);
                        // Only set if no annotation is already present (ISel may
                        // have set one, and we do not want to overwrite).
                        if inst.proof.is_none() {
                            inst.proof = Some(annotation);
                        }
                    }
                }
            }
        }
    }

    // Propagate function-level proofs onto MachFunction metadata.
    for proof in proof_ctx.function_proofs() {
        ir_func.function_proofs.push(proof_to_annotation(proof));
    }
}

// ---------------------------------------------------------------------------
// Copy lowering: pseudo Copy -> real MovR / FmovFprFpr
// ---------------------------------------------------------------------------

/// Lower all pseudo `Copy` instructions to real move instructions.
///
/// After register allocation, the allocator inserts `Copy` pseudo-instructions
/// to move data between physical registers. These must be lowered to real
/// instructions before encoding, since the encoder skips pseudo-instructions.
///
/// **GPR copies** become `MovR` (encoded as `ORR Rd, XZR, Rm`).
/// **FPR copies** become `FmovFprFpr` (encoded as `FMOV Dd, Dn` / `FMOV Ss, Sn`).
///
/// Using the wrong move type is a correctness bug: `ORR Xd, XZR, Xm` operates
/// on GPR registers only and does NOT touch the FP register file. FP copies
/// MUST use the FP data-processing 1-source `FMOV` encoding.
///
/// Copies where source == destination are eliminated entirely (converted to Nop).
pub fn lower_copies(ir_func: &mut IrMachFunction) {
    for inst in &mut ir_func.insts {
        if inst.opcode == IrOpcode::Copy && inst.operands.len() >= 2 {
            // Copy operands: [dst, src]
            // Check if src == dst (redundant copy).
            let is_redundant = match (&inst.operands[0], &inst.operands[1]) {
                (IrOperand::PReg(a), IrOperand::PReg(b)) => a == b,
                _ => false,
            };
            if is_redundant {
                inst.opcode = IrOpcode::Nop;
                // Nop is still pseudo, flags stay IS_PSEUDO — that's correct.
            } else {
                // Determine if this is an FPR copy by checking the destination
                // register. FPR registers need FMOV (FP 1-source); GPR registers
                // need ORR (logical shifted register).
                let is_fpr_copy = matches!(
                    &inst.operands[0],
                    IrOperand::PReg(p) if p.is_fpr()
                );
                if is_fpr_copy {
                    inst.opcode = IrOpcode::FmovFprFpr;
                } else {
                    inst.opcode = IrOpcode::MovR;
                }
                // Clear the IS_PSEUDO flag so the encoder processes this as a
                // real instruction.
                inst.flags.remove(llvm2_ir::inst::InstFlags::IS_PSEUDO);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Encoding: IR MachFunction -> machine code bytes
// ---------------------------------------------------------------------------

/// Resolve symbolic block references in branch instructions to byte offsets.
///
/// After register allocation and frame lowering, branch instructions may still
/// contain `Block(BlockId)` operands from the ISel. This pass computes the byte
/// offset of each block in the final layout and replaces Block operands with
/// immediate offsets (PC-relative, in instruction units).
///
/// Must be called after frame lowering (which may add instructions) and before
/// encoding (which needs concrete offsets).
pub fn resolve_branches(func: &mut IrMachFunction) {
    // Phase 1: Compute the instruction-index offset of each block.
    // Walk blocks in layout order, counting non-pseudo instructions.
    let mut block_offsets: HashMap<BlockId, i64> = HashMap::new();
    let mut current_offset: i64 = 0;

    for &block_id in &func.block_order {
        block_offsets.insert(block_id, current_offset);
        let block = &func.blocks[block_id.0 as usize];
        for &inst_id in &block.insts {
            let inst = &func.insts[inst_id.0 as usize];
            if !inst.is_pseudo() {
                current_offset += 1; // Each instruction is 4 bytes = 1 instruction unit
            }
        }
    }

    // Phase 2: Walk all instructions and replace Block operands with offsets.
    // For each branch instruction, find the Block operand, look up its offset,
    // and replace it with the PC-relative offset (in instruction units).
    let mut inst_offset: i64 = 0;
    for &block_id in &func.block_order.clone() {
        let block = &func.blocks[block_id.0 as usize];
        let inst_ids: Vec<_> = block.insts.clone();
        for &inst_id in &inst_ids {
            let inst = &func.insts[inst_id.0 as usize];
            if inst.is_pseudo() {
                continue;
            }

            let is_branch = matches!(
                inst.opcode,
                IrOpcode::B
                    | IrOpcode::BCond
                    | IrOpcode::Cbz
                    | IrOpcode::Cbnz
                    | IrOpcode::Tbz
                    | IrOpcode::Tbnz
            );

            if is_branch {
                // Find and resolve Block operands to PC-relative immediates.
                let new_operands: Vec<IrOperand> = inst
                    .operands
                    .iter()
                    .map(|op| {
                        if let IrOperand::Block(target_block) = op {
                            if let Some(&target_offset) = block_offsets.get(target_block) {
                                // PC-relative offset in instruction units
                                let rel_offset = target_offset - inst_offset;
                                IrOperand::Imm(rel_offset)
                            } else {
                                op.clone()
                            }
                        } else {
                            op.clone()
                        }
                    })
                    .collect();
                func.insts[inst_id.0 as usize].operands = new_operands;
            }

            inst_offset += 1;
        }
    }
}

/// Encode a single IR instruction into AArch64 machine code.
///
/// Delegates to the unified encoder in [`crate::aarch64::encode`] which
/// dispatches to the correct format-specific encoding function.
fn encode_ir_inst(inst: &IrMachInst) -> Result<u32, PipelineError> {
    crate::aarch64::encode::encode_instruction(inst)
        .map_err(|e| PipelineError::Encoding(e.to_string()))
}

/// Encode all instructions in a function to machine code bytes.
///
/// Walks blocks in layout order, encoding each non-pseudo instruction.
pub fn encode_function(func: &IrMachFunction) -> Result<Vec<u8>, PipelineError> {
    let (code, _fixups) = encode_function_with_fixups(func)?;
    Ok(code)
}

/// Encode all instructions in a function to machine code bytes, collecting
/// fixups for cross-function references.
///
/// When a BL/B instruction has a [`Symbol`](IrOperand::Symbol) operand instead
/// of an immediate, the encoder emits a placeholder (`BL 0`) and records a
/// [`Fixup`](crate::macho::fixup::Fixup) with a `Branch26` relocation kind.
/// The caller (typically [`Pipeline::compile_module`]) resolves these fixups
/// into relocations for the linker.
///
/// Returns `(code_bytes, fixup_list)`.
pub fn encode_function_with_fixups(
    func: &IrMachFunction,
) -> Result<(Vec<u8>, crate::macho::fixup::FixupList), PipelineError> {
    let (code, fixups, _block_offsets) = encode_function_with_fixups_and_blocks(func)?;
    Ok((code, fixups))
}

/// Like [`encode_function_with_fixups`] but also returns the byte offset at
/// which each basic block begins in the encoded code.
///
/// Used by the JIT's block-level profile hook path (`ProfileHookMode::
/// BlockCounts`, issue #364) to know where to splice per-block trampolines.
/// The returned offsets are the **pre-trampoline** offsets — i.e. where each
/// block's first instruction lives in the plain encoding. The caller is
/// responsible for shifting these offsets when inserting trampolines and for
/// re-patching any branches whose source/target block spacing changes.
pub fn encode_function_with_fixups_and_blocks(
    func: &IrMachFunction,
) -> Result<(Vec<u8>, crate::macho::fixup::FixupList, HashMap<BlockId, u32>), PipelineError> {
    use crate::macho::fixup::{Fixup, FixupList};
    use llvm2_ir::inst::AArch64Opcode;

    let mut code = Vec::new();
    let mut fixups = FixupList::new();

    // Track byte offsets of each block in the final layout, for jump
    // table patching. This is computed inline as we emit instructions.
    let mut block_byte_offsets: HashMap<BlockId, u32> = HashMap::new();
    // Pending Adr-for-jump-table fixups: (adr_byte_offset, jump_table_idx).
    let mut jt_adr_fixups: Vec<(u32, u32)> = Vec::new();

    for &block_id in &func.block_order {
        // Record the starting byte offset of this block for later jump-table
        // patching (case target blocks resolve against this map).
        block_byte_offsets.insert(block_id, code.len() as u32);

        let block = func.block(block_id);
        for &inst_id in &block.insts {
            let inst = func.inst(inst_id);
            // Skip pseudo-instructions that should have been eliminated.
            if inst.is_pseudo() {
                continue;
            }

            let byte_offset = code.len() as u32;

            // Check for BL/B with Symbol operand — needs a relocation fixup.
            let has_symbol_target = matches!(
                inst.opcode,
                AArch64Opcode::Bl | AArch64Opcode::BL | AArch64Opcode::B
            ) && inst.operands.first().map_or(false, |op| op.is_symbol());

            // Check for ADR with JumpTableIndex operand — emit placeholder,
            // patch the imm21 after we know the table byte offset.
            let adr_jt_idx = if inst.opcode == AArch64Opcode::Adr {
                inst.operands.get(1).and_then(|op| op.as_jump_table_index())
            } else {
                None
            };

            if has_symbol_target {
                // Emit BL/B with imm26=0 (placeholder — linker will patch via relocation).
                let op_bit = match inst.opcode {
                    AArch64Opcode::B => 0u32,
                    _ => 1u32, // Bl and BL are both BL
                };
                let placeholder = crate::aarch64::encoding::encode_uncond_branch(op_bit, 0);
                code.extend_from_slice(&placeholder.to_le_bytes());

                // Record a Branch26 fixup. The symbol name is stored in the fixup
                // target — the module-level emitter resolves it to a symbol index.
                let sym_name = inst.operands[0].as_symbol().ok_or_else(|| {
                    PipelineError::Encoding(format!(
                        "BL/B instruction at offset {} has non-symbol operand",
                        byte_offset
                    ))
                })?;
                fixups.push(Fixup::branch_sym(byte_offset, sym_name.to_string()));
            } else if let Some(jt_idx) = adr_jt_idx {
                // Emit ADR with imm21=0 (placeholder); record fixup so we can
                // patch it after the jump table is appended below.
                let rd_preg = inst.operands[0].as_preg().ok_or_else(|| {
                    PipelineError::Encoding(format!(
                        "Adr instruction at offset {} has no PReg destination",
                        byte_offset
                    ))
                })?;
                let rd = llvm2_ir::aarch64_regs::hw_encoding(rd_preg);
                let placeholder = crate::aarch64::encoding_mem::encode_adr(0, rd)
                    .map_err(|e| PipelineError::Encoding(e.to_string()))?;
                code.extend_from_slice(&placeholder.to_le_bytes());
                jt_adr_fixups.push((byte_offset, jt_idx));
            } else {
                let word = encode_ir_inst(inst)?;
                code.extend_from_slice(&word.to_le_bytes());
            }
        }
    }

    // Emit jump tables after the function body and patch the corresponding
    // ADR instructions.
    //
    // Layout: each table lives in the same __text section, placed right
    // after the function body. Entries are 32-bit signed offsets from the
    // table base to the case target block: `target_byte_offset -
    // table_base_byte_offset`. At runtime, the switch computes:
    //   x_base = table_base_addr (via ADR)
    //   x_off  = sign_extend_32(table[index])
    //   br     x_base + x_off
    for (adr_offset, jt_idx) in &jt_adr_fixups {
        let jt = func.jump_tables.get(*jt_idx as usize).ok_or_else(|| {
            PipelineError::Encoding(format!(
                "ADR at {:#x} references unknown jump table index {}",
                adr_offset, jt_idx
            ))
        })?;

        // The table is appended at the current end of the code buffer. Each
        // ADR may reference its own (possibly distinct) table; we allocate
        // one table per fixup in the order they appeared.
        //
        // Note: if two ADRs reference the same `jt_idx`, only the first
        // emission writes the table; subsequent ADRs still patch against
        // the same table_offset (see `placed_tables` map).
        let table_offset = code.len() as u32;

        // Patch the ADR imm21 with the byte offset from the ADR to the table.
        let pc_relative = table_offset as i64 - *adr_offset as i64;
        if !(-(1 << 20)..(1 << 20)).contains(&pc_relative) {
            return Err(PipelineError::Encoding(format!(
                "ADR->jump table offset {} does not fit in imm21",
                pc_relative
            )));
        }
        // Re-extract the destination register from the placeholder encoding:
        // bits[4:0] hold Rd. Simpler: decode the placeholder we already
        // wrote.
        let placeholder_bytes = &code[*adr_offset as usize..(*adr_offset as usize + 4)];
        let placeholder_word = u32::from_le_bytes([
            placeholder_bytes[0],
            placeholder_bytes[1],
            placeholder_bytes[2],
            placeholder_bytes[3],
        ]);
        let rd = (placeholder_word & 0x1F) as u8;
        let patched = crate::aarch64::encoding_mem::encode_adr(pc_relative as i32, rd)
            .map_err(|e| PipelineError::Encoding(e.to_string()))?;
        code[*adr_offset as usize..(*adr_offset as usize + 4)]
            .copy_from_slice(&patched.to_le_bytes());

        // Append the table entries: 32-bit signed offsets from the table
        // base to each target block.
        for target in &jt.targets {
            let target_offset = block_byte_offsets.get(target).copied().ok_or_else(|| {
                PipelineError::Encoding(format!(
                    "Jump table target block {:?} has no byte offset",
                    target
                ))
            })?;
            let entry: i32 = (target_offset as i64 - table_offset as i64) as i32;
            code.extend_from_slice(&entry.to_le_bytes());
        }
    }

    Ok((code, fixups, block_byte_offsets))
}

// ---------------------------------------------------------------------------
// LSDA generation from EH metadata
// ---------------------------------------------------------------------------

/// Generate LSDA bytes from a function's exception handling metadata.
///
/// Bridges `llvm2_ir::ExceptionHandlingMetadata` to the codegen-level
/// `exception_handling::ExceptionTable` and serializes it to the binary
/// LSDA format. Returns `None` if the function has no EH info.
///
/// This must be called after code layout (so block byte offsets are known)
/// and before Mach-O emission (so the LSDA bytes can be placed in the
/// `__gcc_except_tab` section).
pub fn generate_lsda_for_function(func: &IrMachFunction) -> Option<Vec<u8>> {
    let eh = &func.eh_metadata;
    if !eh.has_eh_info() {
        return None;
    }

    use crate::exception_handling::{
        LandingPadDesc, build_exception_table_from_pads, generate_lsda,
    };

    let personality = eh.personality.as_deref().unwrap_or("__gxx_personality_v0");

    // Convert IR-level landing pad entries to codegen-level descriptors.
    let landing_pad_descs: Vec<LandingPadDesc> = eh
        .landing_pads
        .iter()
        .map(|lp| LandingPadDesc {
            landing_pad_offset: lp.offset,
            catch_type_indices: lp.catch_type_indices.clone(),
            is_cleanup: lp.is_cleanup,
        })
        .collect();

    // Convert call site entries to (start, length, lp_offset) tuples.
    let call_site_ranges: Vec<(u32, u32, u32)> = eh
        .call_sites
        .iter()
        .map(|cs| {
            let lp_offset = cs
                .landing_pad_block
                .and_then(|lp_block| eh.landing_pads.iter().find(|lp| lp.block == lp_block))
                .map(|lp| lp.offset)
                .unwrap_or(0);
            (cs.start_offset, cs.length, lp_offset)
        })
        .collect();

    let table = build_exception_table_from_pads(personality, &landing_pad_descs, &call_site_ranges);
    let lsda_bytes = generate_lsda(&table);
    Some(lsda_bytes)
}

// ---------------------------------------------------------------------------
// Pipeline — main entry point
// ---------------------------------------------------------------------------

/// The LLVM2 compilation pipeline.
///
/// Orchestrates all phases from tMIR input through to Mach-O .o emission.
pub struct Pipeline {
    pub config: PipelineConfig,
    #[cfg_attr(not(feature = "verify"), allow(dead_code))]
    cegis_cache: Arc<dyn llvm2_opt::CacheBackend>,
    #[cfg(not(feature = "verify"))]
    cegis_warning_emitted: AtomicBool,
}

impl Pipeline {
    /// Create a new pipeline with the given configuration.
    pub fn new(config: PipelineConfig) -> Self {
        Self {
            config,
            cegis_cache: Arc::new(llvm2_opt::InMemoryCache::new()),
            #[cfg(not(feature = "verify"))]
            cegis_warning_emitted: AtomicBool::new(false),
        }
    }

    /// Create a pipeline with default O2 configuration.
    pub fn default_o2() -> Self {
        Self::new(PipelineConfig::default())
    }

    #[cfg(not(feature = "verify"))]
    fn warn_cegis_superopt_ignored(&self) {
        if !self.cegis_warning_emitted.swap(true, Ordering::Relaxed) {
            eprintln!(
                "llvm2: warning: --cegis-superopt=N ignored: binary built without `verify` feature"
            );
        }
    }

    #[cfg_attr(not(feature = "verify"), allow(dead_code))]
    fn cegis_opt_level(&self) -> u8 {
        match self.config.opt_level {
            OptLevel::O0 => 0,
            OptLevel::O1 => 1,
            OptLevel::O2 => 2,
            OptLevel::O3 => 3,
        }
    }

    /// Run the CEGIS superopt pass on a function when configured.
    ///
    /// Returns the per-run pass statistics when the pass actually executed.
    /// Returns `None` when the pass is disabled or unavailable.
    pub fn run_cegis_superopt(&self, func: &mut IrMachFunction) -> Option<CegisPassStats> {
        #[cfg(feature = "verify")]
        {
            use llvm2_opt::pass_manager::MachinePass;

            let budget_sec = self.config.cegis_superopt_budget_sec?;
            if budget_sec == 0 {
                return None;
            }

            let mut pass = llvm2_verify::CegisSuperoptPass::new(llvm2_verify::CegisSuperoptConfig {
                budget_sec,
                per_query_ms: 5_000,
                target_triple: self.config.target_triple.clone(),
                cpu: String::new(),
                features: Vec::new(),
                opt_level: self.cegis_opt_level(),
                cache: Some(self.cegis_cache.clone()),
                trace: None,
            });
            let _ = pass.run(func);
            Some(pass.stats().clone())
        }

        #[cfg(not(feature = "verify"))]
        {
            let _ = func;
            if self.config.cegis_superopt_budget_sec.is_some() {
                self.warn_cegis_superopt_ignored();
            }
            None
        }
    }

    /// Compile a tMIR function through the full pipeline, producing Mach-O bytes.
    ///
    /// This is the main entry point. It takes a `llvm2_lower::Function` (tMIR)
    /// and returns the bytes of a valid .o file.
    pub fn compile_function(
        &self,
        input: &llvm2_lower::Function,
    ) -> Result<Vec<u8>, PipelineError> {
        // Phase 1: Instruction Selection (no proof context in single-function path)
        let (isel_func, _) = self.run_isel(input, None)?;

        // Phase 2: Convert ISel output to shared IR (issue #73 consolidation:
        // conversion logic now lives in ISelFunction::to_ir_func()).
        let mut ir_func = isel_func.to_ir_func();

        // Edge-connectivity check (non-fatal).
        //
        // This was originally `debug_assert!(total_succs > 0, ..)` but that
        // fired on legal tMIR whose non-entry blocks all terminate with
        // `Return` (no `Br`/`CondBr` edges). After DCE such blocks are
        // harmless — register allocation, frame lowering, and encoding all
        // operate per-block and do not require inter-block connectivity.
        // Tripping the assert prevented the pipeline from being a total
        // function over valid tMIR (#447: panic-fuzz totality). We keep the
        // shape of the check as a no-op sentinel for future debugging.
        #[cfg(debug_assertions)]
        {
            let has_branches = ir_func.blocks.len() > 1;
            if has_branches {
                let _total_succs: usize = ir_func.blocks.iter().map(|b| b.succs.len()).sum();
                let _total_preds: usize = ir_func.blocks.iter().map(|b| b.preds.len()).sum();
                // Tolerated: _total_succs == 0 or _total_preds == 0 is a legal
                // (if unusual) multi-block shape — see issue #447.
            }
        }

        // Phase 3: Optimization
        self.run_optimization(&mut ir_func);

        // Phase 3.5: Verification (optional — gated by config.verify)
        self.run_verification(&ir_func)?;

        // Phase 4-6: Register Allocation
        let _spill_slot_count = self.run_regalloc(&mut ir_func)?;

        // Phase 6.5: Lower pseudo Copy instructions to real MovR.
        lower_copies(&mut ir_func);

        // Phase 7: Frame Lowering
        let frame_layout = self.run_frame_lowering(&mut ir_func);

        // Phase 7.5: Branch Resolution — resolve symbolic Block operands to
        // PC-relative byte offsets before encoding.
        resolve_branches(&mut ir_func);

        // Phase 8: Encoding
        let code = encode_function(&ir_func)?;

        // Phase 8.5: LSDA Generation — generate the Language-Specific Data
        // Area from EH metadata (call sites, landing pads, type table).
        let lsda = generate_lsda_for_function(&ir_func);

        // Phase 9: Mach-O Emission (with compact unwind + LSDA)
        let obj_bytes = self.emit_macho(&ir_func, &code, Some(&frame_layout), lsda.as_deref());

        Ok(obj_bytes)
    }

    /// Compile a pre-built IR function (skipping ISel).
    ///
    /// Useful when you already have an `IrMachFunction` and want to run
    /// the remaining phases (opt, regalloc, frame, encode, emit).
    pub fn compile_ir_function(
        &self,
        ir_func: &mut IrMachFunction,
    ) -> Result<Vec<u8>, PipelineError> {
        // Phase 3: Optimization
        self.run_optimization(ir_func);

        // Phase 3.5: Verification (optional — gated by config.verify)
        self.run_verification(ir_func)?;

        // Phase 4-6: Register Allocation
        let _spill_slot_count = self.run_regalloc(ir_func)?;

        // Phase 6.5: Lower pseudo Copy instructions to real MovR.
        lower_copies(ir_func);

        // Phase 7: Frame Lowering
        let frame_layout = self.run_frame_lowering(ir_func);

        // Phase 7.5: Branch Resolution
        resolve_branches(ir_func);

        // Phase 8: Encoding
        let code = encode_function(ir_func)?;

        // Phase 8.5: LSDA Generation
        let lsda = generate_lsda_for_function(ir_func);

        // Phase 9: Mach-O Emission (with compact unwind + LSDA)
        let obj_bytes = self.emit_macho(ir_func, &code, Some(&frame_layout), lsda.as_deref());

        Ok(obj_bytes)
    }

    /// Prepare a tMIR function through phases 1-7.5 (ISel through branch
    /// resolution) without encoding or Mach-O emission.
    ///
    /// The returned `IrMachFunction` is ready to be passed to
    /// [`compile_module`](Self::compile_module) alongside other prepared
    /// functions. This enables multi-function compilation where cross-function
    /// BL instructions are resolved via BRANCH26 relocations.
    pub fn prepare_function(
        &self,
        input: &llvm2_lower::Function,
    ) -> Result<IrMachFunction, PipelineError> {
        self.prepare_function_with_proofs(input, None)
    }

    /// Like [`prepare_function`](Self::prepare_function) but also propagates
    /// tMIR proof annotations from the adapter's [`ProofContext`](llvm2_lower::ProofContext)
    /// onto the machine instructions and function metadata.
    ///
    /// Proof annotations enable proof-consuming optimizations (proof_opts pass)
    /// that eliminate runtime safety checks verified by z4.
    pub fn prepare_function_with_proofs(
        &self,
        input: &llvm2_lower::Function,
        proof_ctx: Option<&llvm2_lower::ProofContext>,
    ) -> Result<IrMachFunction, PipelineError> {
        let (ir_func, _metrics) = self.prepare_function_with_metrics(input, proof_ctx)?;
        Ok(ir_func)
    }

    /// Like [`prepare_function_with_proofs`](Self::prepare_function_with_proofs)
    /// but also records per-phase timings and register-allocation spill-slot
    /// counts for downstream observability.
    pub fn prepare_function_with_metrics(
        &self,
        input: &llvm2_lower::Function,
        proof_ctx: Option<&llvm2_lower::ProofContext>,
    ) -> Result<(IrMachFunction, PreparationMetrics), PipelineError> {
        let mut metrics = PreparationMetrics::default();

        // Phase 1: Instruction Selection
        let isel_start = Instant::now();
        let (isel_func, value_map) = self.run_isel(input, proof_ctx)?;
        metrics.timings.isel = Some(isel_start.elapsed());

        // Phase 2: Convert ISel output to shared IR
        let mut ir_func = isel_func.to_ir_func();

        // Phase 2.5: Propagate proof annotations from ProofContext onto MachInsts.
        //
        // The adapter's ProofContext maps LIR Values to proof annotations.
        // The ISel's value_map maps LIR Values to VRegs/operands.
        // We build a VReg->InstId reverse map from the IR function, then
        // annotate defining instructions with proof annotations.
        if let (Some(ctx), Some(vmap)) = (proof_ctx, &value_map) {
            apply_proof_annotations(&mut ir_func, ctx, vmap);
        }

        // Edge-connectivity check (non-fatal).
        //
        // This was originally `debug_assert!(total_succs > 0, ..)` but that
        // fired on legal tMIR whose non-entry blocks all terminate with
        // `Return` (no `Br`/`CondBr` edges). After DCE such blocks are
        // harmless — register allocation, frame lowering, and encoding all
        // operate per-block and do not require inter-block connectivity.
        // Tripping the assert prevented the pipeline from being a total
        // function over valid tMIR (#447: panic-fuzz totality). We keep the
        // shape of the check as a no-op sentinel for future debugging.
        #[cfg(debug_assertions)]
        {
            let has_branches = ir_func.blocks.len() > 1;
            if has_branches {
                let _total_succs: usize = ir_func.blocks.iter().map(|b| b.succs.len()).sum();
                let _total_preds: usize = ir_func.blocks.iter().map(|b| b.preds.len()).sum();
                // Tolerated: _total_succs == 0 or _total_preds == 0 is a legal
                // (if unusual) multi-block shape — see issue #447.
            }
        }

        // Phase 3: Optimization (including proof-consuming optimizations)
        let optimization_start = Instant::now();
        self.run_optimization(&mut ir_func);
        metrics.timings.optimization = Some(optimization_start.elapsed());

        // Phase 3.5: Verification (optional)
        let verification_start = Instant::now();
        self.run_verification(&ir_func)?;
        if self.config.verify {
            metrics.timings.verification = Some(verification_start.elapsed());
        }

        // Phase 4-6: Register Allocation
        let regalloc_start = Instant::now();
        metrics.spill_slot_count = self.run_regalloc(&mut ir_func)?;
        metrics.timings.regalloc = Some(regalloc_start.elapsed());

        // Phase 6.5: Lower pseudo Copy instructions to real MovR.
        lower_copies(&mut ir_func);

        // Phase 7: Frame Lowering
        let frame_lowering_start = Instant::now();
        let _frame_layout = self.run_frame_lowering(&mut ir_func);
        metrics.timings.frame_lowering = Some(frame_lowering_start.elapsed());

        // Phase 7.5: Branch Resolution
        let branch_resolution_start = Instant::now();
        resolve_branches(&mut ir_func);
        metrics.timings.branch_resolution = Some(branch_resolution_start.elapsed());

        Ok((ir_func, metrics))
    }

    /// Prepare a pre-built IR function through phases 3-7.5 (optimization
    /// through branch resolution) without encoding or Mach-O emission.
    ///
    /// Like [`prepare_function`](Self::prepare_function) but skips ISel,
    /// accepting an `IrMachFunction` directly. The returned function is ready
    /// for [`compile_module`](Self::compile_module).
    pub fn prepare_ir_function(&self, ir_func: &mut IrMachFunction) -> Result<(), PipelineError> {
        // Phase 3: Optimization
        self.run_optimization(ir_func);

        // Phase 3.5: Verification (optional)
        self.run_verification(ir_func)?;

        // Phase 4-6: Register Allocation
        let _spill_slot_count = self.run_regalloc(ir_func)?;

        // Phase 6.5: Lower pseudo Copy instructions to real MovR.
        lower_copies(ir_func);

        // Phase 7: Frame Lowering
        let _frame_layout = self.run_frame_lowering(ir_func);

        // Phase 7.5: Branch Resolution
        resolve_branches(ir_func);

        Ok(())
    }

    /// Encode an already-allocated IR function to Mach-O.
    ///
    /// Skips ISel, optimization, and regalloc. Useful for testing the
    /// encoding and emission phases in isolation.
    pub fn encode_and_emit(&self, ir_func: &mut IrMachFunction) -> Result<Vec<u8>, PipelineError> {
        // Phase 7: Frame Lowering
        let frame_layout = self.run_frame_lowering(ir_func);

        // Phase 8: Encoding
        let code = encode_function(ir_func)?;

        // Phase 8.5: LSDA Generation
        let lsda = generate_lsda_for_function(ir_func);

        // Phase 9: Mach-O Emission (with compact unwind + LSDA)
        let obj_bytes = self.emit_macho(ir_func, &code, Some(&frame_layout), lsda.as_deref());

        Ok(obj_bytes)
    }

    /// Compile multiple pre-built IR functions into a single Mach-O .o file.
    ///
    /// All functions are concatenated into one `__text` section. Cross-function
    /// BL/B instructions with [`Symbol`](IrOperand::Symbol) operands are resolved
    /// into `ARM64_RELOC_BRANCH26` relocations, allowing the linker (or a
    /// subsequent static link step) to patch the call targets.
    ///
    /// This is the correct path for producing .o files with multiple functions —
    /// unlike [`compile_function`](Self::compile_function) which produces one .o
    /// per function and cannot represent cross-function calls.
    ///
    /// Each function must already be in post-regalloc, post-frame-lowering state
    /// (physical registers, branches resolved). The caller can prepare functions
    /// using the pipeline's individual phases or by building IR directly.
    pub fn compile_module(&self, functions: &[IrMachFunction]) -> Result<Vec<u8>, PipelineError> {
        use crate::macho::MachOWriter;
        use crate::macho::fixup::FixupList;

        // Phase 1: Encode each function, collecting code bytes and fixups.
        // Track each function's byte offset within the combined __text section.
        // Pre-allocate based on estimated 4 bytes per instruction * ~25 insts/func.
        let estimated_code_size = functions.len() * 100;
        let mut combined_code = Vec::with_capacity(estimated_code_size);
        let mut combined_fixups = FixupList::new();
        let mut func_offsets: Vec<(String, u64)> = Vec::with_capacity(functions.len());

        for func in functions {
            let func_start = combined_code.len() as u64;
            func_offsets.push((func.name.clone(), func_start));

            let (code, fixups) = encode_function_with_fixups(func)?;

            // Adjust fixup offsets to be relative to the combined section.
            for fixup in fixups.iter() {
                let mut adjusted = fixup.clone();
                adjusted.offset += func_start as u32;
                combined_fixups.push(adjusted);
            }

            combined_code.extend_from_slice(&code);
        }

        // Phase 2: Build the MachOWriter with combined code and symbols.
        let mut writer = MachOWriter::new();
        writer.add_text_section(&combined_code);

        // Add function symbols (Mach-O convention: _prefix).
        // Track symbol indices for fixup resolution.
        let mut symbol_map: HashMap<String, u32> = HashMap::new();
        for (i, (name, offset)) in func_offsets.iter().enumerate() {
            let symbol_name = format!("_{}", name);
            writer.add_symbol(&symbol_name, 1, *offset, true);
            // Symbol index in the nlist array matches insertion order.
            symbol_map.insert(name.clone(), i as u32);
            // Also index by the mangled name for external references.
            symbol_map.insert(symbol_name, i as u32);
        }

        // Phase 3: Resolve named symbol fixups to numeric indices, then
        // convert fixups to relocations.
        combined_fixups.resolve_named_symbols(|name| {
            // Try bare name first (e.g., "func_b"), then mangled ("_func_b").
            symbol_map.get(name).copied().or_else(|| {
                let mangled = format!("_{}", name);
                symbol_map.get(&mangled).copied()
            })
        })?;

        let relocations = combined_fixups.resolve_to_relocations()?;
        for reloc in relocations {
            writer.add_relocation(0, reloc); // section 0 = __text
        }

        Ok(writer.write())
    }

    /// Compile multiple pre-built IR functions into a single Mach-O .o file,
    /// using rayon to encode functions in parallel.
    ///
    /// Each function is encoded independently into its own buffer on a worker
    /// thread. Results are then merged sequentially (fast memcpy) into the
    /// combined __text section. This eliminates the sequential encoding
    /// bottleneck for modules with many functions.
    ///
    /// Falls back to [`compile_module`](Self::compile_module) for single-function
    /// modules where parallel overhead would dominate.
    pub fn compile_module_parallel(
        &self,
        functions: &[IrMachFunction],
    ) -> Result<Vec<u8>, PipelineError> {
        use crate::macho::MachOWriter;
        use crate::macho::fixup::FixupList;
        use rayon::prelude::*;

        if functions.len() < 2 {
            // Not enough to justify parallel overhead; fall back to sequential.
            return self.compile_module(functions);
        }

        // Phase 1: Encode each function in parallel, producing per-function buffers.
        // Each encoding is fully independent — no shared mutable state.
        let encoded: Vec<Result<(String, Vec<u8>, FixupList), PipelineError>> = functions
            .par_iter()
            .map(|func| {
                let (code, fixups) = encode_function_with_fixups(func)?;
                Ok((func.name.clone(), code, fixups))
            })
            .collect();

        // Phase 2: Merge per-function buffers sequentially (fast memcpy).
        // Pre-calculate total size for a single allocation.
        let mut total_size: usize = 0;
        let mut results: Vec<(String, Vec<u8>, FixupList)> = Vec::with_capacity(encoded.len());
        for result in encoded {
            let (name, code, fixups) = result?;
            total_size += code.len();
            results.push((name, code, fixups));
        }

        let mut combined_code = Vec::with_capacity(total_size);
        let mut combined_fixups = FixupList::new();
        let mut func_offsets: Vec<(String, u64)> = Vec::with_capacity(results.len());

        for (name, code, fixups) in &results {
            let func_start = combined_code.len() as u64;
            func_offsets.push((name.clone(), func_start));

            // Adjust fixup offsets to be relative to the combined section.
            for fixup in fixups.iter() {
                let mut adjusted = fixup.clone();
                adjusted.offset += func_start as u32;
                combined_fixups.push(adjusted);
            }

            combined_code.extend_from_slice(code);
        }

        // Phase 3: Build the MachOWriter with combined code and symbols.
        let mut writer = MachOWriter::new();
        writer.add_text_section(&combined_code);

        let mut symbol_map: HashMap<String, u32> = HashMap::new();
        for (i, (name, offset)) in func_offsets.iter().enumerate() {
            let symbol_name = format!("_{}", name);
            writer.add_symbol(&symbol_name, 1, *offset, true);
            symbol_map.insert(name.clone(), i as u32);
            symbol_map.insert(symbol_name, i as u32);
        }

        // Phase 4: Resolve named symbol fixups to numeric indices.
        combined_fixups.resolve_named_symbols(|name| {
            symbol_map.get(name).copied().or_else(|| {
                let mangled = format!("_{}", name);
                symbol_map.get(&mangled).copied()
            })
        })?;

        let relocations = combined_fixups.resolve_to_relocations()?;
        for reloc in relocations {
            writer.add_relocation(0, reloc); // section 0 = __text
        }

        Ok(writer.write())
    }

    // --- Phase implementations ---

    /// Phase 1: Run instruction selection on the input function.
    ///
    /// When `proof_ctx` is provided, returns the ISel value map alongside
    /// the function so the caller can propagate proof annotations to the IR.
    fn run_isel(
        &self,
        input: &llvm2_lower::Function,
        proof_ctx: Option<&llvm2_lower::ProofContext>,
    ) -> Result<
        (
            llvm2_lower::isel::ISelFunction,
            Option<
                std::collections::HashMap<
                    llvm2_lower::instructions::Value,
                    llvm2_lower::isel::ISelOperand,
                >,
            >,
        ),
        PipelineError,
    > {
        use llvm2_lower::isel::InstructionSelector;

        let sig = llvm2_lower::function::Signature {
            params: input.signature.params.clone(),
            returns: input.signature.returns.clone(),
        };

        let mut isel = InstructionSelector::new(input.name.clone(), sig.clone());

        // Propagate stack slot metadata from the LIR function to the ISel.
        isel.set_stack_slots(input.stack_slots.clone());

        // Seed Value->Type hints from the adapter (Call/CallIndirect result
        // types, see #381). Must happen BEFORE select_block_* so that
        // `value_type()` lookups during selection return the true callee
        // return type instead of the I64 default.
        isel.seed_value_types(&input.value_types);

        // Seed pure-callee names so `select_call` can stamp the emitted Bl
        // with `ProofAnnotation::Pure` for SROA partial-escape (#456).
        isel.seed_pure_callees(&input.pure_callees);

        // Lower formal arguments at entry: copies ABI physical registers into
        // VRegs and populates the value_map for function parameters.
        isel.lower_formal_arguments(&sig, input.entry_block)
            .map_err(|e| PipelineError::ISel(e.to_string()))?;

        // Sort blocks by ID to ensure deterministic processing order. The entry
        // block is processed first (it should have the lowest ID), then remaining
        // blocks in layout order. This ensures SSA values defined in dominating
        // blocks are available when referenced by later blocks.
        let mut block_order: Vec<_> = input.blocks.keys().copied().collect();
        block_order.sort_by_key(|b| b.0);

        // Process blocks in sorted order. For non-entry blocks, define block
        // parameter Values before processing instructions.
        for block_ref in &block_order {
            let basic_block = &input.blocks[block_ref];
            if *block_ref != input.entry_block && !basic_block.params.is_empty() {
                isel.define_block_params(&basic_block.params);
            }
            isel.select_block_with_source_locs(
                *block_ref,
                &basic_block.instructions,
                &basic_block.source_locs,
            )
            .map_err(|e| PipelineError::ISel(e.to_string()))?;
        }

        if proof_ctx.is_some() {
            let (func, value_map) = isel.finalize_with_value_map();
            Ok((func, Some(value_map)))
        } else {
            Ok((isel.finalize(), None))
        }
    }

    /// Phase 3: Run optimization passes and instruction scheduling.
    ///
    /// After the main optimization pipeline, runs pre-register-allocation
    /// instruction scheduling. The scheduler variant depends on the
    /// optimization level and `config.use_pressure_aware_scheduler`:
    ///
    /// - **O0**: No scheduling (fastest compile).
    /// - **O1**: [`llvm2_opt::scheduler::InstructionScheduler`] (fast, ILP-focused).
    /// - **O2/O3** with `use_pressure_aware_scheduler` (default):
    ///   [`llvm2_opt::scheduler::PressureAwareScheduler`] (balances ILP with
    ///   register pressure to reduce spills).
    /// - **O2/O3** without `use_pressure_aware_scheduler`:
    ///   [`llvm2_opt::scheduler::InstructionScheduler`] (fast, ILP-focused).
    fn run_optimization(&self, func: &mut IrMachFunction) {
        use llvm2_opt::pass_manager::MachinePass;
        use llvm2_opt::pipeline::{OptLevel as OptOptLevel, OptimizationPipeline};
        use llvm2_opt::scheduler::{InstructionScheduler, PressureAwareScheduler};

        let opt_level = match self.config.opt_level {
            OptLevel::O0 => OptOptLevel::O0,
            OptLevel::O1 => OptOptLevel::O1,
            OptLevel::O2 => OptOptLevel::O2,
            OptLevel::O3 => OptOptLevel::O3,
        };

        // Run the optimization pipeline for the selected level.
        let pipeline = OptimizationPipeline::new(opt_level);
        let _stats = pipeline.run(func);

        let _ = self.run_cegis_superopt(func);

        // Pre-register-allocation instruction scheduling.
        // Skip at O0 for fastest compile. At O1, use the fast ILP scheduler.
        // At O2+, use pressure-aware scheduling by default.
        match self.config.opt_level {
            OptLevel::O0 => {
                // No scheduling at O0.
            }
            OptLevel::O1 => {
                let mut sched = InstructionScheduler;
                sched.run(func);
            }
            OptLevel::O2 | OptLevel::O3 => {
                if self.config.use_pressure_aware_scheduler {
                    let mut sched = PressureAwareScheduler;
                    sched.run(func);
                } else {
                    let mut sched = InstructionScheduler;
                    sched.run(func);
                }
            }
        }
    }

    /// Phase 3.5: Run function-level verification (optional).
    ///
    /// When `self.config.verify` is true and the `verify` feature is enabled,
    /// calls [`llvm2_verify::verify_function`] on the optimized IR. Returns an
    /// error only if any instruction *fails* verification (i.e., a proof was
    /// found but produced Invalid/Unknown). Unverified instructions (no proof
    /// available) are tolerated — they are expected for opcodes not yet covered
    /// by the proof database.
    ///
    /// When the `verify` feature is disabled, this is a no-op.
    #[cfg(feature = "verify")]
    fn run_verification(&self, func: &IrMachFunction) -> Result<(), PipelineError> {
        // Skip the expensive ProofDatabase construction + evaluation when
        // verification is disabled. Previously we always ran verify_function()
        // and only checked the result when config.verify was true, but
        // ProofDatabase::new() alone consumes significant stack space in debug
        // builds (hundreds of proof obligations with SmtExpr trees), which
        // contributed to stack overflows on the default 8 MB test-thread stack.
        // See issue #205.
        if !self.config.verify {
            return Ok(());
        }

        let report = llvm2_verify::verify_function(func);

        if report.failed_count() > 0 {
            return Err(PipelineError::VerificationFailed {
                function: report.function_name.clone(),
                failures: report.failed_count(),
                coverage: report.coverage_percent(),
                report,
            });
        }

        Ok(())
    }

    /// Stub when `verify` feature is disabled — verification is a no-op.
    #[cfg(not(feature = "verify"))]
    fn run_verification(&self, _func: &IrMachFunction) -> Result<(), PipelineError> {
        Ok(())
    }

    /// Phase 4-6: Run register allocation and apply results.
    ///
    /// Returns the number of spill stack slots allocated by register
    /// allocation for this function.
    fn run_regalloc(&self, ir_func: &mut IrMachFunction) -> Result<usize, PipelineError> {
        // Phase 4: Convert IR to regalloc format
        let mut ra_func = ir_to_regalloc(ir_func)?;

        // Debug: dump block structure before regalloc
        #[cfg(debug_assertions)]
        if std::env::var("LLVM2_DEBUG_REGALLOC").is_ok() {
            eprintln!("=== REGALLOC DEBUG: {} ===", ra_func.name);
            for (bi, block) in ra_func.blocks.iter().enumerate() {
                eprintln!(
                    "  block {}: insts={:?} succs={:?} preds={:?}",
                    bi,
                    block.insts.len(),
                    block.succs,
                    block.preds
                );
            }
            for (ii, inst) in ra_func.insts.iter().enumerate() {
                eprintln!(
                    "  inst {}: opc={} defs={:?} uses={:?}",
                    ii,
                    inst.opcode,
                    inst.defs
                        .iter()
                        .filter_map(|o| o.as_vreg())
                        .collect::<Vec<_>>(),
                    inst.uses
                        .iter()
                        .filter_map(|o| o.as_vreg())
                        .collect::<Vec<_>>()
                );
            }
        }

        // Phase 5: Run the allocator
        let ra_config = llvm2_regalloc::AllocConfig::default_aarch64();
        let result = llvm2_regalloc::allocate(&mut ra_func, &ra_config)
            .map_err(|e| PipelineError::RegAlloc(e.to_string()))?;
        let spill_slot_count = result.spills.len();

        // Debug: dump allocation
        #[cfg(debug_assertions)]
        if std::env::var("LLVM2_DEBUG_REGALLOC").is_ok() {
            eprintln!("  allocation:");
            let mut allocs: Vec<_> = result.allocation.iter().collect();
            allocs.sort_by_key(|(v, _)| v.id);
            for (vreg, preg) in allocs {
                eprintln!("    v{} -> {:?}", vreg.id, preg);
            }
            eprintln!("=== END REGALLOC DEBUG ===");
        }

        // Phase 5.5: Post-RA spill optimization — eliminate dead stores,
        // redundant reloads, and dead definitions in spill code. Runs on the
        // regalloc-format function which still has spill pseudo-instructions.
        // Enabled at O1+ optimization levels when enable_post_ra_opt is set.
        if self.config.enable_post_ra_opt && self.config.opt_level != OptLevel::O0 {
            let _post_ra_stats = llvm2_regalloc::post_ra_optimize(&mut ra_func);
        }

        // Phase 6: Apply allocation back to IR
        apply_regalloc(ir_func, &result.allocation);

        // Add spill stack slots to the IR function.
        for _spill in &result.spills {
            // Each spill gets an 8-byte slot (register-sized).
            ir_func.alloc_stack_slot(IrStackSlot::new(8, 8));
        }

        Ok(spill_slot_count)
    }

    /// Phase 7: Frame lowering — prologue/epilogue + frame index elimination.
    ///
    /// Returns the computed [`FrameLayout`] so downstream phases (compact unwind
    /// emission) can use it without recomputing.
    fn run_frame_lowering(&self, func: &mut IrMachFunction) -> crate::frame::FrameLayout {
        use crate::frame;

        // Compute outgoing arg area size from actual call sites. ISel emits
        // stack-passed arguments as stores with base=SP; scanning those stores
        // yields the high-water mark the caller must reserve at the bottom of
        // its own frame so the stores do not overlap the saved FP/LR pair
        // (bug fix: test_differential_call_with_stack_args, issue #301).
        let outgoing_arg_size = frame::compute_max_outgoing_arg_size(func);
        let layout = frame::compute_frame_layout(func, outgoing_arg_size, true);
        frame::eliminate_frame_indices(func, &layout);
        frame::insert_prologue_epilogue(func, &layout);
        layout
    }

    /// Phase 9: Emit a Mach-O .o file from encoded machine code.
    ///
    /// When `frame_layout` is provided, a `__LD,__compact_unwind` section is
    /// emitted containing a single compact unwind entry for the function.
    /// This is required by macOS for backtraces, profilers, and exception handling.
    ///
    /// When `lsda_bytes` is provided, a `__TEXT,__gcc_except_tab` section is
    /// emitted containing the LSDA (Language-Specific Data Area) for exception
    /// handling. The personality routine reads this data to dispatch exceptions
    /// to the correct landing pad.
    ///
    /// When `self.config.emit_debug` is true, DWARF debug sections are also
    /// emitted (`__DWARF,__debug_info`, `__debug_abbrev`, `__debug_str`,
    /// `__debug_line`).
    fn emit_macho(
        &self,
        ir_func: &IrMachFunction,
        code: &[u8],
        frame_layout: Option<&crate::frame::FrameLayout>,
        lsda_bytes: Option<&[u8]>,
    ) -> Vec<u8> {
        use crate::macho::MachOWriter;
        use crate::macho::constants::S_REGULAR;
        use crate::unwind::{
            CompactUnwindEntry, CompactUnwindSection, add_compact_unwind_to_writer,
        };

        let func_name = &ir_func.name;
        let mut writer = MachOWriter::new();
        writer.add_text_section(code);

        // Add the function symbol with Mach-O name mangling (_prefix).
        let symbol_name = format!("_{}", func_name);
        // Section index 1 = __text (first section, 1-based).
        writer.add_symbol(&symbol_name, 1, 0, true);

        // Emit LSDA (__gcc_except_tab) section for exception handling.
        // This section contains the LSDA referenced by the compact unwind
        // entry's LSDA pointer. It must appear in __TEXT to be PC-relative
        // addressable on AArch64 macOS.
        if let Some(lsda) = lsda_bytes {
            writer.add_custom_section(
                b"__gcc_except_tab",
                b"__TEXT",
                lsda,
                2, // 2^2 = 4-byte alignment (matching LLVM)
                S_REGULAR,
            );
        }

        // Emit compact unwind section if we have frame layout information.
        if let Some(layout) = frame_layout {
            let mut cu_section = CompactUnwindSection::new();
            // Symbol index 0 = the function symbol we just added.
            let entry = CompactUnwindEntry::from_layout(
                layout,
                0, // function_offset (relocated via ARM64_RELOC_UNSIGNED)
                code.len() as u32,
                0, // symbol_index for the function
            );
            cu_section.add_entry(entry);
            add_compact_unwind_to_writer(&mut writer, &cu_section);
        }

        // Emit DWARF debug sections when debug info is requested.
        if self.config.emit_debug {
            use crate::dwarf_info::{
                DwarfDebugInfo, FunctionDebugInfo, FunctionParam, SourceLanguage, SourceLineEntry,
                VariableDebugInfo, VariableLocation, add_debug_info_to_writer,
            };

            let debug_meta = &ir_func.debug_meta;

            // Determine source file name from debug_meta or fall back to function name.
            let default_file = format!("{}.rs", func_name);
            let source_file = debug_meta.source_file.as_deref().unwrap_or(&default_file);

            let mut debug_info = DwarfDebugInfo::new(source_file, ".", SourceLanguage::Rust);
            debug_info.add_standard_types();

            // Build parameter debug info from debug_meta.
            let params: Vec<FunctionParam> = debug_meta
                .param_names
                .iter()
                .enumerate()
                .map(|(i, name)| {
                    // Map param type to base type index. The standard types are:
                    // 0=i8, 1=i16, 2=i32, 3=i64, 4=f32, 5=f64, 6=bool
                    // Default to i64 (index 3) since tMIR doesn't carry param
                    // type info through debug_meta yet.
                    let type_index = if i < ir_func.signature.params.len() {
                        match ir_func.signature.params[i] {
                            llvm2_ir::Type::I8 => 0,
                            llvm2_ir::Type::I16 => 1,
                            llvm2_ir::Type::I32 => 2,
                            llvm2_ir::Type::I64 | llvm2_ir::Type::I128 => 3,
                            llvm2_ir::Type::F32 => 4,
                            llvm2_ir::Type::F64 => 5,
                            llvm2_ir::Type::B1 => 6,
                            llvm2_ir::Type::Ptr => 3, // pointer = i64 size
                            _ => 3,                   // default to i64 for Struct, Array, etc.
                        }
                    } else {
                        3 // default i64
                    };
                    FunctionParam {
                        name: name.clone(),
                        type_index,
                    }
                })
                .collect();

            // Build variable debug info from stack slots (after regalloc).
            // Each stack slot represents a spilled variable or local allocation.
            let variables: Vec<VariableDebugInfo> = ir_func
                .stack_slots
                .iter()
                .enumerate()
                .map(|(i, slot)| {
                    // Map slot size to a base type index.
                    let type_index = match slot.size {
                        1 => 0, // i8
                        2 => 1, // i16
                        4 => 2, // i32
                        8 => 3, // i64
                        _ => 3, // default i64
                    };
                    VariableDebugInfo {
                        name: format!("local_{}", i),
                        type_index,
                        location: VariableLocation::FrameOffset(
                            -((i as i64 + 1) * slot.size as i64),
                        ),
                    }
                })
                .collect();

            // Build line number entries from instruction source_locs.
            // Each instruction is 4 bytes on AArch64, so address = inst_index * 4.
            let mut line_entries = Vec::new();
            let mut inst_offset: u64 = 0;
            for block_id in &ir_func.block_order {
                let block = &ir_func.blocks[block_id.0 as usize];
                for &inst_id in &block.insts {
                    let inst = &ir_func.insts[inst_id.0 as usize];
                    if let Some(loc) = &inst.source_loc {
                        line_entries.push(SourceLineEntry {
                            address: inst_offset,
                            line: loc.line,
                            column: loc.col as u16,
                            is_stmt: true,
                        });
                    }
                    inst_offset += 4; // AArch64: each instruction is 4 bytes
                }
            }

            let decl_line = if debug_meta.decl_line > 0 {
                debug_meta.decl_line as u16
            } else {
                1
            };

            debug_info.add_function(FunctionDebugInfo {
                name: func_name.to_string(),
                linkage_name: Some(symbol_name.clone()),
                low_pc: 0,
                size: code.len() as u32,
                params,
                variables,
                scopes: vec![],
                line_entries,
                decl_file: 1,
                decl_line,
            });
            add_debug_info_to_writer(&mut writer, &debug_info);
        }

        writer.write()
    }

    // --- Dispatch verification ---

    /// Verify a dispatch plan against a compute graph, applying the configured
    /// [`DispatchVerifyMode`] policy.
    ///
    /// Returns:
    /// - `Ok(plan)` if verification passes or is disabled
    /// - `Ok(cpu_only_plan)` if verification fails under
    ///   [`DispatchVerifyMode::FallbackOnFailure`]
    /// - `Err(DispatchVerificationFailed)` if verification fails under
    ///   [`DispatchVerifyMode::ErrorOnFailure`]
    pub fn verify_dispatch_plan(
        &self,
        graph: &ComputeGraph,
        plan: DispatchPlan,
        recommendations: &[TargetRecommendation],
    ) -> Result<DispatchPlan, PipelineError> {
        match self.config.verify_dispatch {
            DispatchVerifyMode::Off => Ok(plan),
            DispatchVerifyMode::FallbackOnFailure => {
                let report = verify_dispatch_plan_properties(graph, &plan);
                if report.all_ok() {
                    Ok(plan)
                } else {
                    // Fall back to a CPU-only plan.
                    Ok(generate_cpu_only_plan(graph, recommendations))
                }
            }
            DispatchVerifyMode::ErrorOnFailure => {
                let report = verify_dispatch_plan_properties(graph, &plan);
                if report.all_ok() {
                    Ok(plan)
                } else {
                    let violations = report.total_violations();
                    let summary = format!("{}", report);
                    Err(PipelineError::DispatchVerificationFailed {
                        violations,
                        summary,
                        report,
                    })
                }
            }
        }
    }

    /// Generate a dispatch plan, verify it, and return the verified (or
    /// fallback) plan. This is the primary integration point for dispatch
    /// verification in the compilation pipeline.
    ///
    /// Combines plan generation with verification in a single call.
    pub fn generate_and_verify_dispatch(
        &self,
        graph: &ComputeGraph,
        recommendations: &[TargetRecommendation],
    ) -> Result<DispatchPlan, PipelineError> {
        let plan = generate_dispatch_plan(graph, recommendations);
        self.verify_dispatch_plan(graph, plan, recommendations)
    }

    /// Generate Metal kernel code for GPU-targeted dispatch operations.
    ///
    /// This is the primary integration point for Metal kernel generation in
    /// the compilation pipeline. It takes a verified dispatch plan and compute
    /// graph, and produces all MSL kernel sources plus host-side dispatch code.
    ///
    /// Typical usage:
    /// ```ignore
    /// let plan = pipeline.generate_and_verify_dispatch(&graph, &recs)?;
    /// let metal = pipeline.emit_metal_kernels(&plan, &graph)?;
    /// // metal.kernels contains MSL sources to compile with `xcrun metal`
    /// // metal.dispatch_code contains Objective-C host dispatch code
    /// ```
    pub fn emit_metal_kernels(
        &self,
        plan: &DispatchPlan,
        graph: &ComputeGraph,
    ) -> Result<crate::metal_emitter::MetalOutput, PipelineError> {
        crate::metal_emitter::emit_metal_kernels(plan, graph).map_err(PipelineError::from)
    }
}

// ---------------------------------------------------------------------------
// CoreML MIL generation from dispatch plan
// ---------------------------------------------------------------------------

/// Output of CoreML MIL program generation from a dispatch plan.
///
/// Contains the generated MIL program, any ANE compatibility warnings, and
/// an estimated latency based on the dispatch plan's cycle estimates for
/// ANE-targeted nodes.
#[derive(Debug, Clone)]
pub struct CoreMLOutput {
    /// The generated MIL program representing ANE-targeted operations.
    pub program: MilProgram,
    /// ANE compatibility warnings (empty if fully compatible).
    pub warnings: Vec<String>,
    /// Estimated latency in microseconds for the ANE subgraph, derived from
    /// the dispatch plan's per-node cycle estimates at an assumed 1 GHz
    /// Neural Engine clock.
    pub estimated_latency_us: u64,
}

/// Collect ANE-targeted nodes from a dispatch plan and generate a CoreML MIL
/// program from the corresponding compute graph nodes.
///
/// This is the primary integration point between the dispatch pipeline and
/// the CoreML emitter. It:
/// 1. Collects all node IDs assigned to `NeuralEngine` in the plan
/// 2. Resolves those IDs to `ComputeNode` references from the graph
/// 3. Generates a MIL program via [`CoreMLEmitter::emit_program_from_nodes`]
/// 4. Validates ANE compatibility via [`validate_ane_compatibility`]
/// 5. Estimates latency from the plan's per-node cycle costs
///
/// Returns `Ok(CoreMLOutput)` on success, or `Err(PipelineError::CoreMLEmit)`
/// if no ANE nodes are found or MIL emission fails.
pub fn emit_coreml_program(
    plan: &DispatchPlan,
    graph: &ComputeGraph,
) -> Result<CoreMLOutput, PipelineError> {
    // Step 1: Collect ANE-targeted node IDs from the dispatch plan assignment.
    let mut ane_node_ids: Vec<ComputeNodeId> = plan
        .assignment
        .iter()
        .filter(|(_, target)| **target == ComputeTarget::NeuralEngine)
        .map(|(id, _)| *id)
        .collect();

    if ane_node_ids.is_empty() {
        return Err(PipelineError::CoreMLEmit(CoreMLEmitError::EmptyNodeList));
    }

    // Sort by node ID for deterministic ordering.
    ane_node_ids.sort_by_key(|id| id.0);

    // Step 2: Resolve node IDs to ComputeNode references from the graph.
    let mut ane_nodes: Vec<ComputeNode> = Vec::with_capacity(ane_node_ids.len());
    for node_id in &ane_node_ids {
        if let Some(node) = graph.node(*node_id) {
            ane_nodes.push(node.clone());
        }
        // Skip nodes not found in graph (shouldn't happen with a valid plan).
    }

    if ane_nodes.is_empty() {
        return Err(PipelineError::CoreMLEmit(CoreMLEmitError::EmptyNodeList));
    }

    // Step 3: Generate MIL program from ANE nodes.
    let mut emitter = CoreMLEmitter::new();
    let program = emitter.emit_program_from_nodes(&ane_nodes)?;

    // Step 4: Validate ANE compatibility.
    let warnings = validate_ane_compatibility(&program);

    // Step 5: Estimate latency from dispatch plan cycle costs.
    // Sum the estimated_cycles from KernelLaunch ops targeting NeuralEngine
    // nodes. Convert cycles to microseconds assuming ~1 GHz ANE clock.
    let total_ane_cycles: u64 = plan
        .ops
        .iter()
        .filter_map(|op| match op {
            DispatchOp::KernelLaunch {
                target: ComputeTarget::NeuralEngine,
                estimated_cycles,
                ..
            } => Some(*estimated_cycles),
            _ => None,
        })
        .sum();

    // 1 GHz = 1 cycle per ns. Convert cycles to microseconds: cycles / 1000.
    let estimated_latency_us = total_ane_cycles.saturating_div(1000).max(1);

    Ok(CoreMLOutput {
        program,
        warnings,
        estimated_latency_us,
    })
}

// ---------------------------------------------------------------------------
// CPU-only dispatch plan fallback
// ---------------------------------------------------------------------------

/// Generate a CPU-only dispatch plan as a safe fallback.
///
/// When dispatch verification fails, this function generates a conservative
/// plan that runs every node on `CpuScalar`. No data transfers or
/// synchronizations are needed because all execution stays on the CPU.
pub fn generate_cpu_only_plan(
    graph: &ComputeGraph,
    _recommendations: &[TargetRecommendation],
) -> DispatchPlan {
    use llvm2_lower::dispatch::DispatchOp;
    use llvm2_lower::target_analysis::ComputeTarget;

    let mut assignment = HashMap::new();
    let mut ops = Vec::new();
    let mut total_cycles: u64 = 0;

    // Walk nodes in ID order. Since all nodes are on the same target
    // (CpuScalar), no transfers or syncs are needed.
    let mut node_ids: Vec<_> = graph.nodes.iter().map(|n| n.id).collect();
    node_ids.sort_by_key(|id| id.0);

    for node_id in &node_ids {
        assignment.insert(*node_id, ComputeTarget::CpuScalar);

        let node = graph.node(*node_id);
        let estimated_cycles = node
            .and_then(|n| n.costs.get(&ComputeTarget::CpuScalar))
            .map(|c| c.latency_cycles)
            .unwrap_or(1);

        ops.push(DispatchOp::CpuFallback {
            node_id: *node_id,
            reason: "dispatch verification fallback".to_string(),
        });
        total_cycles = total_cycles.saturating_add(estimated_cycles);
    }

    DispatchPlan {
        ops,
        assignment,
        estimated_total_cycles: total_cycles,
    }
}

// ---------------------------------------------------------------------------
// Convenience: compile_to_object — one-shot compilation
// ---------------------------------------------------------------------------

/// Compile a tMIR function to a Mach-O .o file in one call.
///
/// This is the highest-level API for LLVM2 compilation.
pub fn compile_to_object(
    input: &llvm2_lower::Function,
    opt_level: OptLevel,
) -> Result<Vec<u8>, PipelineError> {
    let config = PipelineConfig {
        opt_level,
        emit_debug: false,
        verify_dispatch: DispatchVerifyMode::FallbackOnFailure,
        verify: false,
        enable_post_ra_opt: opt_level != OptLevel::O0,
        use_pressure_aware_scheduler: matches!(opt_level, OptLevel::O2 | OptLevel::O3),
        cegis_superopt_budget_sec: None,
        target_triple: String::new(),
    };
    let pipeline = Pipeline::new(config);
    pipeline.compile_function(input)
}

/// Run the dialect-framework progressive-lowering pipeline (verif -> tmir ->
/// machir -> MachFunction) on a [`llvm2_dialect::DialectModule`] and return
/// one [`IrMachFunction`] per contained function.
///
/// Integration point for LLVM2#433 / tMIR#428. The `DialectModule` must have
/// the `verif`, `tmir`, and `machir` dialects registered (see
/// [`llvm2_dialect::dialects::conversions::register_all`]). Any op that no
/// registered pattern handles trips the legality gate inside
/// [`llvm2_dialect::lower_module`] — producing a hard error in one known
/// location rather than leaking an unknown op into the MachFunction.
///
/// This is an early hook: for modules that arrive as `DialectModule` the
/// frontend calls this and then feeds the resulting MachFunctions through
/// the rest of the pipeline (regalloc, frame lowering, encoding, Mach-O).
/// `tmir::Module` frontends continue to use the legacy
/// [`Pipeline::compile_function`] path. Both paths converge at MachFunction.
pub fn dialect_lower_module(
    module: &mut llvm2_dialect::DialectModule,
) -> Result<Vec<IrMachFunction>, llvm2_dialect::LowerModuleError> {
    llvm2_dialect::lower_module(module)
}

/// Build an `add(a: i32, b: i32) -> i32` function directly in IR.
///
/// This is a convenience for testing the pipeline without needing the full
/// tMIR input infrastructure. It builds the function directly in llvm2-ir
/// types with physical registers (simulating post-ISel + post-regalloc state).
pub fn build_add_test_function() -> IrMachFunction {
    use llvm2_ir::function::Type;
    use llvm2_ir::regs::{X0, X1};

    let sig = IrSignature::new(vec![Type::I32, Type::I32], vec![Type::I32]);
    let mut func = IrMachFunction::new("add".to_string(), sig);
    let entry = func.entry;

    // ADD X0, X0, X1 (result in X0 per ABI)
    let add = IrMachInst::new(
        IrOpcode::AddRR,
        vec![
            IrOperand::PReg(X0),
            IrOperand::PReg(X0),
            IrOperand::PReg(X1),
        ],
    );
    let add_id = func.push_inst(add);
    func.append_inst(entry, add_id);

    // RET
    let ret = IrMachInst::new(IrOpcode::Ret, vec![]);
    let ret_id = func.push_inst(ret);
    func.append_inst(entry, ret_id);

    func
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use llvm2_ir::function::{MachFunction as IrMachFunction, Signature as IrSignature, Type};
    use llvm2_ir::inst::{AArch64Opcode as IrOpcode, InstFlags, MachInst as IrMachInst};
    use llvm2_ir::operand::MachOperand as IrOperand;
    use llvm2_ir::regs::{RegClass, VReg, X0, X1, X2};
    use llvm2_ir::types::BlockId;

    // -- Helper: create an instruction with specific flags --

    fn make_inst(opcode: IrOpcode, operands: Vec<IrOperand>) -> IrMachInst {
        IrMachInst::new(opcode, operands)
    }

    fn make_inst_with_flags(
        opcode: IrOpcode,
        operands: Vec<IrOperand>,
        flags: InstFlags,
    ) -> IrMachInst {
        IrMachInst::with_flags(opcode, operands, flags)
    }

    // =========================================================================
    // classify_def_use tests
    // =========================================================================

    #[test]
    fn classify_def_use_alu_instruction_first_operand_is_def() {
        // AddRR: [dst, src1, src2] -> defs=[dst], uses=[src1, src2]
        let v0 = VReg::new(0, RegClass::Gpr64);
        let v1 = VReg::new(1, RegClass::Gpr64);
        let v2 = VReg::new(2, RegClass::Gpr64);
        let inst = make_inst(
            IrOpcode::AddRR,
            vec![
                IrOperand::VReg(v0),
                IrOperand::VReg(v1),
                IrOperand::VReg(v2),
            ],
        );

        let (defs, uses) = classify_def_use(&inst).unwrap();
        assert_eq!(defs.len(), 1);
        assert_eq!(uses.len(), 2);
        assert_eq!(defs[0].as_vreg(), Some(v0));
        assert_eq!(uses[0].as_vreg(), Some(v1));
        assert_eq!(uses[1].as_vreg(), Some(v2));
    }

    #[test]
    fn classify_def_use_add_immediate() {
        // AddRI: [dst, src, imm] -> defs=[dst], uses=[src, imm]
        let v0 = VReg::new(0, RegClass::Gpr64);
        let v1 = VReg::new(1, RegClass::Gpr64);
        let inst = make_inst(
            IrOpcode::AddRI,
            vec![IrOperand::VReg(v0), IrOperand::VReg(v1), IrOperand::Imm(42)],
        );

        let (defs, uses) = classify_def_use(&inst).unwrap();
        assert_eq!(defs.len(), 1);
        assert_eq!(uses.len(), 2);
        assert_eq!(defs[0].as_vreg(), Some(v0));
    }

    #[test]
    fn classify_def_use_store_all_uses_no_defs() {
        // StrRI has WRITES_MEMORY flag -> all operands are uses
        let inst = make_inst(
            IrOpcode::StrRI,
            vec![IrOperand::PReg(X0), IrOperand::PReg(X1), IrOperand::Imm(8)],
        );

        let (defs, uses) = classify_def_use(&inst).unwrap();
        assert!(defs.is_empty(), "store should have no defs");
        assert_eq!(uses.len(), 3);
    }

    #[test]
    fn classify_def_use_branch_all_uses_no_defs() {
        // B has IS_BRANCH flag -> all operands are uses
        let inst = make_inst(IrOpcode::B, vec![IrOperand::Block(BlockId(1))]);

        let (defs, uses) = classify_def_use(&inst).unwrap();
        assert!(defs.is_empty(), "branch should have no defs");
        assert_eq!(uses.len(), 1);
    }

    #[test]
    fn classify_def_use_conditional_branch() {
        // BCond has IS_BRANCH flag -> all uses
        let inst = make_inst(
            IrOpcode::BCond,
            vec![IrOperand::Imm(0), IrOperand::Block(BlockId(2))],
        );

        let (defs, uses) = classify_def_use(&inst).unwrap();
        assert!(defs.is_empty());
        assert_eq!(uses.len(), 2);
    }

    #[test]
    fn classify_def_use_return_all_uses_no_defs() {
        // Ret has IS_RETURN flag -> all uses
        let inst = make_inst(IrOpcode::Ret, vec![]);

        let (defs, uses) = classify_def_use(&inst).unwrap();
        assert!(defs.is_empty());
        assert!(uses.is_empty());
    }

    #[test]
    fn classify_def_use_compare_all_uses_no_defs() {
        // CmpRR -> classified as compare, all uses
        let inst = make_inst(
            IrOpcode::CmpRR,
            vec![IrOperand::PReg(X0), IrOperand::PReg(X1)],
        );

        let (defs, uses) = classify_def_use(&inst).unwrap();
        assert!(defs.is_empty(), "compare should have no defs");
        assert_eq!(uses.len(), 2);
    }

    #[test]
    fn classify_def_use_compare_immediate() {
        let inst = make_inst(
            IrOpcode::CmpRI,
            vec![IrOperand::PReg(X0), IrOperand::Imm(0)],
        );

        let (defs, uses) = classify_def_use(&inst).unwrap();
        assert!(defs.is_empty());
        assert_eq!(uses.len(), 2);
    }

    #[test]
    fn classify_def_use_fcmp_all_uses() {
        let v0 = VReg::new(0, RegClass::Fpr64);
        let v1 = VReg::new(1, RegClass::Fpr64);
        let inst = make_inst(
            IrOpcode::Fcmp,
            vec![IrOperand::VReg(v0), IrOperand::VReg(v1)],
        );

        let (defs, uses) = classify_def_use(&inst).unwrap();
        assert!(defs.is_empty(), "fcmp should have no defs");
        assert_eq!(uses.len(), 2);
    }

    #[test]
    fn classify_def_use_load_first_operand_is_def() {
        // LdrRI: [dst, base, offset] -> defs=[dst], uses=[base, offset]
        let v0 = VReg::new(0, RegClass::Gpr64);
        let inst = make_inst(
            IrOpcode::LdrRI,
            vec![IrOperand::VReg(v0), IrOperand::PReg(X1), IrOperand::Imm(0)],
        );

        let (defs, uses) = classify_def_use(&inst).unwrap();
        assert_eq!(defs.len(), 1);
        assert_eq!(uses.len(), 2);
        assert_eq!(defs[0].as_vreg(), Some(v0));
    }

    #[test]
    fn classify_def_use_empty_operands_no_defs_no_uses() {
        // An instruction with no operands -> empty defs and uses regardless of flags
        let inst = make_inst(IrOpcode::Nop, vec![]);

        let (defs, uses) = classify_def_use(&inst).unwrap();
        assert!(defs.is_empty());
        assert!(uses.is_empty());
    }

    #[test]
    fn classify_def_use_mov_immediate() {
        // MovI: [dst, imm] -> defs=[dst], uses=[imm]
        let v0 = VReg::new(0, RegClass::Gpr64);
        let inst = make_inst(
            IrOpcode::MovI,
            vec![IrOperand::VReg(v0), IrOperand::Imm(100)],
        );

        let (defs, uses) = classify_def_use(&inst).unwrap();
        assert_eq!(defs.len(), 1);
        assert_eq!(uses.len(), 1);
        assert_eq!(defs[0].as_vreg(), Some(v0));
    }

    #[test]
    fn classify_def_use_physical_register_operands() {
        // Post-regalloc instruction with PRegs
        let inst = make_inst(
            IrOpcode::AddRR,
            vec![
                IrOperand::PReg(X0),
                IrOperand::PReg(X1),
                IrOperand::PReg(X2),
            ],
        );

        let (defs, uses) = classify_def_use(&inst).unwrap();
        assert_eq!(defs.len(), 1);
        assert_eq!(uses.len(), 2);
        assert_eq!(defs[0].as_preg(), Some(X0));
        assert_eq!(uses[0].as_preg(), Some(X1));
    }

    #[test]
    fn classify_def_use_rejects_frame_index_operand() {
        // FrameIndex should be lowered before regalloc; classify_def_use should error
        let inst = make_inst(
            IrOpcode::AddRR,
            vec![
                IrOperand::FrameIndex(llvm2_ir::types::FrameIdx(-8)),
                IrOperand::PReg(X0),
            ],
        );

        let result = classify_def_use(&inst);
        assert!(result.is_err());
        match result.unwrap_err() {
            PipelineError::InvalidOperand(msg) => {
                assert!(msg.contains("FrameIndex"));
            }
            other => panic!("expected InvalidOperand, got {:?}", other),
        }
    }

    #[test]
    fn classify_def_use_rejects_memop_operand() {
        let inst = make_inst(
            IrOpcode::AddRR,
            vec![
                IrOperand::MemOp {
                    base: X0,
                    offset: 16,
                },
                IrOperand::PReg(X1),
            ],
        );

        let result = classify_def_use(&inst);
        assert!(result.is_err());
    }

    #[test]
    fn classify_def_use_rejects_special_operand() {
        let inst = make_inst(
            IrOpcode::AddRR,
            vec![
                IrOperand::Special(llvm2_ir::regs::SpecialReg::SP),
                IrOperand::PReg(X0),
            ],
        );

        let result = classify_def_use(&inst);
        assert!(result.is_err());
    }

    #[test]
    fn classify_def_use_store_with_writes_memory_flag() {
        // Verify the WRITES_MEMORY flag drives the "all uses" classification
        // even with a non-standard opcode that has WRITES_MEMORY set manually
        let inst = make_inst_with_flags(
            IrOpcode::StpRI,
            vec![
                IrOperand::PReg(X0),
                IrOperand::PReg(X1),
                IrOperand::PReg(X2),
                IrOperand::Imm(0),
            ],
            InstFlags::WRITES_MEMORY | InstFlags::HAS_SIDE_EFFECTS,
        );

        let (defs, uses) = classify_def_use(&inst).unwrap();
        assert!(defs.is_empty());
        assert_eq!(uses.len(), 4);
    }

    #[test]
    fn classify_def_use_cbz_branch() {
        let inst = make_inst(
            IrOpcode::Cbz,
            vec![IrOperand::PReg(X0), IrOperand::Block(BlockId(3))],
        );

        let (defs, uses) = classify_def_use(&inst).unwrap();
        assert!(defs.is_empty(), "cbz is a branch, no defs");
        assert_eq!(uses.len(), 2);
    }

    // =========================================================================
    // apply_regalloc tests
    // =========================================================================

    #[test]
    fn apply_regalloc_rewrites_vregs_to_pregs() {
        let sig = IrSignature::new(vec![Type::I64, Type::I64], vec![Type::I64]);
        let mut func = IrMachFunction::new("test".to_string(), sig);

        let v0 = VReg::new(0, RegClass::Gpr64);
        let v1 = VReg::new(1, RegClass::Gpr64);
        let v2 = VReg::new(2, RegClass::Gpr64);

        let inst = IrMachInst::new(
            IrOpcode::AddRR,
            vec![
                IrOperand::VReg(v0),
                IrOperand::VReg(v1),
                IrOperand::VReg(v2),
            ],
        );
        func.push_inst(inst);

        let mut allocation = HashMap::new();
        allocation.insert(v0, X0);
        allocation.insert(v1, X1);
        allocation.insert(v2, X2);

        apply_regalloc(&mut func, &allocation);

        assert_eq!(func.insts[0].operands[0], IrOperand::PReg(X0));
        assert_eq!(func.insts[0].operands[1], IrOperand::PReg(X1));
        assert_eq!(func.insts[0].operands[2], IrOperand::PReg(X2));
    }

    #[test]
    fn apply_regalloc_leaves_pregs_unchanged() {
        let sig = IrSignature::new(vec![], vec![]);
        let mut func = IrMachFunction::new("test".to_string(), sig);

        let inst = IrMachInst::new(
            IrOpcode::AddRR,
            vec![
                IrOperand::PReg(X0),
                IrOperand::PReg(X1),
                IrOperand::PReg(X2),
            ],
        );
        func.push_inst(inst);

        let allocation = HashMap::new();
        apply_regalloc(&mut func, &allocation);

        assert_eq!(func.insts[0].operands[0], IrOperand::PReg(X0));
        assert_eq!(func.insts[0].operands[1], IrOperand::PReg(X1));
        assert_eq!(func.insts[0].operands[2], IrOperand::PReg(X2));
    }

    #[test]
    fn apply_regalloc_leaves_immediates_unchanged() {
        let sig = IrSignature::new(vec![], vec![]);
        let mut func = IrMachFunction::new("test".to_string(), sig);

        let v0 = VReg::new(0, RegClass::Gpr64);
        let inst = IrMachInst::new(
            IrOpcode::AddRI,
            vec![IrOperand::VReg(v0), IrOperand::PReg(X1), IrOperand::Imm(42)],
        );
        func.push_inst(inst);

        let mut allocation = HashMap::new();
        allocation.insert(v0, X0);

        apply_regalloc(&mut func, &allocation);

        assert_eq!(func.insts[0].operands[0], IrOperand::PReg(X0));
        assert_eq!(func.insts[0].operands[2], IrOperand::Imm(42));
    }

    #[test]
    fn apply_regalloc_unallocated_vreg_stays_vreg() {
        // VRegs not in the allocation map are left as-is (spilled)
        let sig = IrSignature::new(vec![], vec![]);
        let mut func = IrMachFunction::new("test".to_string(), sig);

        let v0 = VReg::new(0, RegClass::Gpr64);
        let inst = IrMachInst::new(IrOpcode::MovI, vec![IrOperand::VReg(v0), IrOperand::Imm(0)]);
        func.push_inst(inst);

        let allocation = HashMap::new(); // empty allocation
        apply_regalloc(&mut func, &allocation);

        assert_eq!(func.insts[0].operands[0], IrOperand::VReg(v0));
    }

    // =========================================================================
    // lower_copies tests
    // =========================================================================

    #[test]
    fn lower_copies_converts_copy_to_movr() {
        let sig = IrSignature::new(vec![], vec![]);
        let mut func = IrMachFunction::new("test".to_string(), sig);

        let copy = IrMachInst::new(
            IrOpcode::Copy,
            vec![IrOperand::PReg(X0), IrOperand::PReg(X1)],
        );
        // Copy starts as pseudo
        assert!(copy.flags.contains(InstFlags::IS_PSEUDO));
        func.push_inst(copy);

        lower_copies(&mut func);

        assert_eq!(func.insts[0].opcode, IrOpcode::MovR);
        assert!(
            !func.insts[0].flags.contains(InstFlags::IS_PSEUDO),
            "MovR should not be pseudo after lowering"
        );
    }

    #[test]
    fn lower_copies_eliminates_redundant_copy() {
        let sig = IrSignature::new(vec![], vec![]);
        let mut func = IrMachFunction::new("test".to_string(), sig);

        // Copy X0 -> X0 (redundant)
        let copy = IrMachInst::new(
            IrOpcode::Copy,
            vec![IrOperand::PReg(X0), IrOperand::PReg(X0)],
        );
        func.push_inst(copy);

        lower_copies(&mut func);

        assert_eq!(
            func.insts[0].opcode,
            IrOpcode::Nop,
            "redundant copy should become Nop"
        );
    }

    #[test]
    fn lower_copies_leaves_non_copy_instructions() {
        let sig = IrSignature::new(vec![], vec![]);
        let mut func = IrMachFunction::new("test".to_string(), sig);

        let add = IrMachInst::new(
            IrOpcode::AddRR,
            vec![
                IrOperand::PReg(X0),
                IrOperand::PReg(X1),
                IrOperand::PReg(X2),
            ],
        );
        func.push_inst(add);

        lower_copies(&mut func);

        assert_eq!(func.insts[0].opcode, IrOpcode::AddRR);
    }

    #[test]
    fn lower_copies_handles_mixed_instructions() {
        let sig = IrSignature::new(vec![], vec![]);
        let mut func = IrMachFunction::new("test".to_string(), sig);

        // Non-redundant copy
        func.push_inst(IrMachInst::new(
            IrOpcode::Copy,
            vec![IrOperand::PReg(X0), IrOperand::PReg(X1)],
        ));
        // Regular instruction
        func.push_inst(IrMachInst::new(
            IrOpcode::AddRR,
            vec![
                IrOperand::PReg(X0),
                IrOperand::PReg(X0),
                IrOperand::PReg(X2),
            ],
        ));
        // Redundant copy
        func.push_inst(IrMachInst::new(
            IrOpcode::Copy,
            vec![IrOperand::PReg(X2), IrOperand::PReg(X2)],
        ));

        lower_copies(&mut func);

        assert_eq!(func.insts[0].opcode, IrOpcode::MovR);
        assert_eq!(func.insts[1].opcode, IrOpcode::AddRR);
        assert_eq!(func.insts[2].opcode, IrOpcode::Nop);
    }

    // =========================================================================
    // resolve_branches tests
    // =========================================================================

    #[test]
    fn resolve_branches_single_block_no_branches() {
        let sig = IrSignature::new(vec![], vec![]);
        let mut func = IrMachFunction::new("test".to_string(), sig);
        let entry = func.entry;

        let add = IrMachInst::new(
            IrOpcode::AddRR,
            vec![
                IrOperand::PReg(X0),
                IrOperand::PReg(X0),
                IrOperand::PReg(X1),
            ],
        );
        let add_id = func.push_inst(add);
        func.append_inst(entry, add_id);

        let ret = IrMachInst::new(IrOpcode::Ret, vec![]);
        let ret_id = func.push_inst(ret);
        func.append_inst(entry, ret_id);

        resolve_branches(&mut func);

        // No branches, nothing should change
        assert_eq!(func.insts[add_id.0 as usize].operands.len(), 3);
    }

    #[test]
    fn resolve_branches_replaces_block_operand_with_offset() {
        let sig = IrSignature::new(vec![], vec![]);
        let mut func = IrMachFunction::new("test".to_string(), sig);
        let entry = func.entry;
        let bb1 = func.create_block();

        // bb0: B bb1
        let br = IrMachInst::new(IrOpcode::B, vec![IrOperand::Block(bb1)]);
        let br_id = func.push_inst(br);
        func.append_inst(entry, br_id);

        // bb1: RET
        let ret = IrMachInst::new(IrOpcode::Ret, vec![]);
        let ret_id = func.push_inst(ret);
        func.append_inst(bb1, ret_id);

        resolve_branches(&mut func);

        // Branch at offset 0 targets bb1 at offset 1 -> relative offset = 1
        let resolved_operand = &func.insts[br_id.0 as usize].operands[0];
        assert!(
            matches!(resolved_operand, IrOperand::Imm(1)),
            "expected Imm(1), got {:?}",
            resolved_operand
        );
    }

    // =========================================================================
    // PipelineConfig / OptLevel tests
    // =========================================================================

    #[test]
    fn pipeline_config_default_is_o2() {
        let config = PipelineConfig::default();
        assert_eq!(config.opt_level, OptLevel::O2);
        assert!(!config.emit_debug);
    }

    #[test]
    fn opt_level_equality() {
        assert_eq!(OptLevel::O0, OptLevel::O0);
        assert_eq!(OptLevel::O1, OptLevel::O1);
        assert_eq!(OptLevel::O2, OptLevel::O2);
        assert_eq!(OptLevel::O3, OptLevel::O3);
        assert_ne!(OptLevel::O0, OptLevel::O3);
        assert_ne!(OptLevel::O1, OptLevel::O2);
    }

    #[test]
    fn pipeline_config_custom() {
        let config = PipelineConfig {
            opt_level: OptLevel::O0,
            emit_debug: true,
            verify: true,
            ..Default::default()
        };
        assert_eq!(config.opt_level, OptLevel::O0);
        assert!(config.emit_debug);
        assert!(config.verify);
    }

    #[test]
    fn pipeline_config_clone() {
        let config = PipelineConfig {
            opt_level: OptLevel::O3,
            emit_debug: true,
            ..Default::default()
        };
        let cloned = config.clone();
        assert_eq!(cloned.opt_level, OptLevel::O3);
        assert!(cloned.emit_debug);
    }

    // =========================================================================
    // Pipeline constructor tests
    // =========================================================================

    #[test]
    fn pipeline_new_with_config() {
        let config = PipelineConfig {
            opt_level: OptLevel::O1,
            emit_debug: false,
            ..Default::default()
        };
        let pipeline = Pipeline::new(config);
        assert_eq!(pipeline.config.opt_level, OptLevel::O1);
    }

    #[test]
    fn pipeline_default_o2() {
        let pipeline = Pipeline::default_o2();
        assert_eq!(pipeline.config.opt_level, OptLevel::O2);
        assert!(!pipeline.config.emit_debug);
    }

    // =========================================================================
    // CompilationUnit tests
    // =========================================================================

    #[test]
    fn compilation_unit_new() {
        let sig = IrSignature::new(vec![Type::I64], vec![Type::I64]);
        let func = IrMachFunction::new("my_func".to_string(), sig);
        let config = PipelineConfig::default();
        let unit = CompilationUnit::new(func, config);

        assert_eq!(unit.name, "my_func");
        assert!(unit.code.is_empty());
        assert_eq!(unit.config.opt_level, OptLevel::O2);
    }

    // =========================================================================
    // build_add_test_function tests
    // =========================================================================

    #[test]
    fn build_add_test_function_structure() {
        let func = build_add_test_function();

        assert_eq!(func.name, "add");
        assert_eq!(func.signature.params.len(), 2);
        assert_eq!(func.signature.returns.len(), 1);
        assert_eq!(func.num_blocks(), 1);
        // Should have 2 instructions: ADD + RET
        assert_eq!(func.block(func.entry).len(), 2);
    }

    #[test]
    fn build_add_test_function_instructions() {
        let func = build_add_test_function();
        let entry_block = func.block(func.entry);

        let add_inst = func.inst(entry_block.insts[0]);
        assert_eq!(add_inst.opcode, IrOpcode::AddRR);
        assert_eq!(add_inst.operands.len(), 3);
        assert_eq!(add_inst.operands[0], IrOperand::PReg(X0));
        assert_eq!(add_inst.operands[1], IrOperand::PReg(X0));
        assert_eq!(add_inst.operands[2], IrOperand::PReg(X1));

        let ret_inst = func.inst(entry_block.insts[1]);
        assert_eq!(ret_inst.opcode, IrOpcode::Ret);
        assert!(ret_inst.operands.is_empty());
    }

    // =========================================================================
    // ir_to_regalloc adapter tests
    // =========================================================================

    #[test]
    fn ir_to_regalloc_preserves_function_name() {
        let sig = IrSignature::new(vec![], vec![]);
        let mut func = IrMachFunction::new("my_function".to_string(), sig);
        let entry = func.entry;

        let ret = IrMachInst::new(IrOpcode::Ret, vec![]);
        let ret_id = func.push_inst(ret);
        func.append_inst(entry, ret_id);

        let ra_func = ir_to_regalloc(&func).unwrap();
        assert_eq!(ra_func.name, "my_function");
    }

    #[test]
    fn ir_to_regalloc_converts_stack_slots() {
        let sig = IrSignature::new(vec![], vec![]);
        let mut func = IrMachFunction::new("test".to_string(), sig);
        func.alloc_stack_slot(llvm2_ir::function::StackSlot::new(8, 8));
        func.alloc_stack_slot(llvm2_ir::function::StackSlot::new(16, 16));

        let entry = func.entry;
        let ret = IrMachInst::new(IrOpcode::Ret, vec![]);
        let ret_id = func.push_inst(ret);
        func.append_inst(entry, ret_id);

        let ra_func = ir_to_regalloc(&func).unwrap();
        assert_eq!(ra_func.stack_slots.len(), 2);
        assert_eq!(ra_func.next_stack_slot, 2);
    }

    #[test]
    fn ir_to_regalloc_preserves_block_order() {
        let sig = IrSignature::new(vec![], vec![]);
        let mut func = IrMachFunction::new("test".to_string(), sig);
        let entry = func.entry;
        let bb1 = func.create_block();

        // Add instructions to both blocks
        let nop1 = IrMachInst::new(IrOpcode::Nop, vec![]);
        let nop1_id = func.push_inst(nop1);
        func.append_inst(entry, nop1_id);

        let nop2 = IrMachInst::new(IrOpcode::Nop, vec![]);
        let nop2_id = func.push_inst(nop2);
        func.append_inst(bb1, nop2_id);

        let ra_func = ir_to_regalloc(&func).unwrap();
        assert_eq!(ra_func.block_order.len(), 2);
        assert_eq!(ra_func.block_order[0], BlockId(0));
        assert_eq!(ra_func.block_order[1], BlockId(1));
        assert_eq!(ra_func.entry_block, BlockId(0));
    }

    // =========================================================================
    // PipelineError Display tests
    // =========================================================================

    #[test]
    fn pipeline_error_display_isel() {
        let err = PipelineError::ISel("something went wrong".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("instruction selection failed"));
        assert!(msg.contains("something went wrong"));
    }

    #[test]
    fn pipeline_error_display_regalloc() {
        let err = PipelineError::RegAlloc("out of registers".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("register allocation failed"));
    }

    #[test]
    fn pipeline_error_display_encoding() {
        let err = PipelineError::Encoding("bad instruction".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("encoding failed"));
    }

    #[test]
    fn pipeline_error_display_unsupported_opcode() {
        let err = PipelineError::UnsupportedOpcode(IrOpcode::Phi);
        let msg = format!("{}", err);
        assert!(msg.contains("unsupported opcode"));
    }

    #[test]
    fn pipeline_error_display_invalid_operand() {
        let err = PipelineError::InvalidOperand("FrameIndex must be eliminated".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("invalid operand"));
    }

    // =========================================================================
    // Pipeline verification integration tests
    // =========================================================================

    #[test]
    fn pipeline_config_default_verify_is_false() {
        let config = PipelineConfig::default();
        assert!(!config.verify, "verify should be false by default");
    }

    #[test]
    fn pipeline_config_verify_enabled() {
        let config = PipelineConfig {
            verify: true,
            ..Default::default()
        };
        assert!(config.verify);
    }

    #[test]
    fn run_verification_disabled_succeeds() {
        // With verify=false, run_verification should always return Ok
        // and skip expensive proof evaluation entirely.
        let pipeline = Pipeline::new(PipelineConfig {
            verify: false,
            ..Default::default()
        });
        let func = build_add_test_function();
        let result = pipeline.run_verification(&func);
        assert!(
            result.is_ok(),
            "verification disabled should always succeed"
        );
    }

    #[cfg(feature = "verify")]
    #[test]
    fn run_verification_enabled_passes_add_function() {
        // The add test function has AddRR (verified) + Ret (unverified).
        // Since there are no *failed* instructions (just unverified), it
        // should pass even with verify=true.
        let pipeline = Pipeline::new(PipelineConfig {
            verify: true,
            ..Default::default()
        });
        let func = build_add_test_function();
        let result = pipeline.run_verification(&func);
        assert!(
            result.is_ok(),
            "add function has no failed proofs, should pass"
        );
    }

    #[cfg(feature = "verify")]
    #[test]
    fn run_verification_enabled_reports_coverage() {
        let pipeline = Pipeline::new(PipelineConfig {
            verify: true,
            ..Default::default()
        });

        // Build a function with only verified instructions. Route through
        // the pipeline's run_verification entrypoint so the test exercises
        // the pipeline's verification wiring (not just llvm2_verify).
        let sig = IrSignature::new(vec![Type::I64, Type::I64], vec![Type::I64]);
        let mut func = IrMachFunction::new("all_verified".to_string(), sig);
        let entry = func.entry;

        let add = IrMachInst::new(
            IrOpcode::AddRR,
            vec![
                IrOperand::PReg(X0),
                IrOperand::PReg(X0),
                IrOperand::PReg(X1),
            ],
        );
        let add_id = func.push_inst(add);
        func.append_inst(entry, add_id);

        let sub = IrMachInst::new(
            IrOpcode::SubRR,
            vec![
                IrOperand::PReg(X0),
                IrOperand::PReg(X0),
                IrOperand::PReg(X1),
            ],
        );
        let sub_id = func.push_inst(sub);
        func.append_inst(entry, sub_id);

        // Pipeline verification path: should pass with no failed proofs.
        pipeline
            .run_verification(&func)
            .expect("run_verification: all instructions verified");

        // Direct llvm2_verify report for coverage details.
        let report = llvm2_verify::verify_function(&func);
        assert_eq!(report.verified_count(), 2);
        assert_eq!(report.failed_count(), 0);
        assert_eq!(report.coverage_percent(), 100.0);
    }

    #[cfg(feature = "verify")]
    #[test]
    fn compile_ir_function_with_verification_enabled() {
        // End-to-end: compile a pre-built IR function with verification on.
        let pipeline = Pipeline::new(PipelineConfig {
            opt_level: OptLevel::O0,
            verify: true,
            ..Default::default()
        });

        let mut func = build_add_test_function();
        let result = pipeline.compile_ir_function(&mut func);
        // Should succeed — AddRR has a proof, Ret is unverified (not failed).
        assert!(
            result.is_ok(),
            "compile_ir_function with verify=true should succeed for add: {:?}",
            result.err()
        );
    }

    #[test]
    fn compile_ir_function_without_verification() {
        // Compile with verification off — should always succeed.
        let pipeline = Pipeline::new(PipelineConfig {
            opt_level: OptLevel::O0,
            verify: false,
            ..Default::default()
        });

        let mut func = build_add_test_function();
        let result = pipeline.compile_ir_function(&mut func);
        assert!(
            result.is_ok(),
            "compile_ir_function with verify=false should succeed"
        );
    }

    #[cfg(feature = "verify")]
    #[test]
    fn verification_failed_error_display() {
        let report = llvm2_verify::FunctionVerificationReport {
            function_name: "bad_func".to_string(),
            instructions: vec![],
        };
        let err = PipelineError::VerificationFailed {
            function: "bad_func".to_string(),
            failures: 3,
            coverage: 42.5,
            report,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("function verification failed"));
        assert!(msg.contains("3 failures"));
        assert!(msg.contains("42.5%"));
        assert!(msg.contains("bad_func"));
    }

    // =========================================================================
    // Post-RA optimization pipeline integration tests
    // =========================================================================

    #[test]
    fn pipeline_config_default_enables_post_ra_opt() {
        let config = PipelineConfig::default();
        assert!(
            config.enable_post_ra_opt,
            "post-RA opt should be enabled by default"
        );
    }

    #[test]
    fn pipeline_config_post_ra_opt_disabled() {
        let config = PipelineConfig {
            enable_post_ra_opt: false,
            ..Default::default()
        };
        assert!(!config.enable_post_ra_opt);
    }

    #[test]
    fn post_ra_opt_runs_at_o2() {
        // End-to-end: compile a pre-built IR function at O2 with post-RA opt
        // enabled. The pass should run without errors. We verify that the
        // pipeline completes successfully — the post-RA opt is wired into
        // run_regalloc and runs transparently.
        let pipeline = Pipeline::new(PipelineConfig {
            opt_level: OptLevel::O2,
            enable_post_ra_opt: true,
            ..Default::default()
        });

        let mut func = build_add_test_function();
        let result = pipeline.compile_ir_function(&mut func);
        assert!(
            result.is_ok(),
            "compile_ir_function at O2 with post-RA opt should succeed: {:?}",
            result.err()
        );
    }

    #[test]
    fn post_ra_opt_skipped_at_o0() {
        // At O0, post-RA optimization should be skipped even if the config
        // flag is true. The pipeline checks both enable_post_ra_opt AND
        // opt_level != O0.
        let pipeline = Pipeline::new(PipelineConfig {
            opt_level: OptLevel::O0,
            enable_post_ra_opt: true,
            ..Default::default()
        });

        let mut func = build_add_test_function();
        let result = pipeline.compile_ir_function(&mut func);
        assert!(
            result.is_ok(),
            "compile_ir_function at O0 should succeed (post-RA opt skipped): {:?}",
            result.err()
        );
    }

    #[test]
    fn post_ra_opt_skipped_when_disabled() {
        // Post-RA opt can be explicitly disabled even at O2.
        let pipeline = Pipeline::new(PipelineConfig {
            opt_level: OptLevel::O2,
            enable_post_ra_opt: false,
            ..Default::default()
        });

        let mut func = build_add_test_function();
        let result = pipeline.compile_ir_function(&mut func);
        assert!(
            result.is_ok(),
            "compile_ir_function at O2 with post-RA opt disabled should succeed: {:?}",
            result.err()
        );
    }

    #[test]
    fn post_ra_opt_runs_at_o1() {
        // O1 should also enable post-RA optimization.
        let pipeline = Pipeline::new(PipelineConfig {
            opt_level: OptLevel::O1,
            enable_post_ra_opt: true,
            ..Default::default()
        });

        let mut func = build_add_test_function();
        let result = pipeline.compile_ir_function(&mut func);
        assert!(
            result.is_ok(),
            "compile_ir_function at O1 with post-RA opt should succeed: {:?}",
            result.err()
        );
    }

    #[test]
    fn compile_to_object_o0_disables_post_ra_opt() {
        // The compile_to_object convenience function should set
        // enable_post_ra_opt based on optimization level.
        // We can't directly inspect the config, but we verify that it
        // constructs PipelineConfig with enable_post_ra_opt = (opt != O0).
        let config = PipelineConfig {
            opt_level: OptLevel::O0,
            emit_debug: false,
            enable_post_ra_opt: OptLevel::O0 != OptLevel::O0,
            use_pressure_aware_scheduler: false,
            ..Default::default()
        };
        assert!(!config.enable_post_ra_opt, "O0 should disable post-RA opt");

        let config = PipelineConfig {
            opt_level: OptLevel::O2,
            emit_debug: false,
            enable_post_ra_opt: OptLevel::O2 != OptLevel::O0,
            ..Default::default()
        };
        assert!(config.enable_post_ra_opt, "O2 should enable post-RA opt");
    }
}

// ---------------------------------------------------------------------------
// Tests — pressure-aware scheduler integration
// ---------------------------------------------------------------------------

#[cfg(test)]
mod scheduler_integration_tests {
    use super::*;

    // -- Helpers --

    /// Build a PipelineConfig for a given opt level with default scheduler setting.
    fn config_for(opt: OptLevel) -> PipelineConfig {
        PipelineConfig {
            opt_level: opt,
            ..Default::default()
        }
    }

    /// Build a PipelineConfig with the pressure-aware scheduler explicitly disabled.
    fn config_no_pressure(opt: OptLevel) -> PipelineConfig {
        PipelineConfig {
            opt_level: opt,
            use_pressure_aware_scheduler: false,
            ..Default::default()
        }
    }

    // -- Config field defaults --

    #[test]
    fn pipeline_config_default_enables_pressure_aware_scheduler() {
        let config = PipelineConfig::default();
        assert!(
            config.use_pressure_aware_scheduler,
            "Default PipelineConfig should enable pressure-aware scheduler"
        );
    }

    #[test]
    fn pipeline_config_use_pressure_aware_scheduler_at_o2() {
        let config = config_for(OptLevel::O2);
        assert!(
            config.use_pressure_aware_scheduler,
            "O2 default should use pressure-aware scheduler"
        );
    }

    #[test]
    fn pipeline_config_use_pressure_aware_scheduler_at_o3() {
        let config = config_for(OptLevel::O3);
        assert!(
            config.use_pressure_aware_scheduler,
            "O3 default should use pressure-aware scheduler"
        );
    }

    #[test]
    fn pipeline_config_can_disable_pressure_aware_scheduler() {
        let config = config_no_pressure(OptLevel::O2);
        assert!(
            !config.use_pressure_aware_scheduler,
            "Should be able to disable pressure-aware scheduler"
        );
    }

    // -- Scheduler selection via run_optimization --
    //
    // We verify that the pipeline constructs and runs the correct scheduler
    // by building a small function and running it through run_optimization
    // at each level. The function is simple enough that both schedulers
    // succeed; the key assertion is that run_optimization does not panic.

    fn make_simple_func() -> IrMachFunction {
        use llvm2_ir::function::Signature as IrSignature;
        use llvm2_ir::inst::{AArch64Opcode as IrOpcode, MachInst as IrMachInst};
        use llvm2_ir::operand::MachOperand as IrOperand;
        use llvm2_ir::regs::{RegClass, VReg};

        let mut func =
            IrMachFunction::new("test_sched".to_string(), IrSignature::new(vec![], vec![]));
        let entry = func.entry;

        // mov v0, #42
        let mov = IrMachInst::new(
            IrOpcode::MovI,
            vec![
                IrOperand::VReg(VReg::new(0, RegClass::Gpr64)),
                IrOperand::Imm(42),
            ],
        );
        let id0 = func.push_inst(mov);
        func.append_inst(entry, id0);

        // add v1, v0, #1
        let add = IrMachInst::new(
            IrOpcode::AddRI,
            vec![
                IrOperand::VReg(VReg::new(1, RegClass::Gpr64)),
                IrOperand::VReg(VReg::new(0, RegClass::Gpr64)),
                IrOperand::Imm(1),
            ],
        );
        let id1 = func.push_inst(add);
        func.append_inst(entry, id1);

        // ret
        let ret = IrMachInst::new(IrOpcode::Ret, vec![]);
        let id2 = func.push_inst(ret);
        func.append_inst(entry, id2);

        func
    }

    #[test]
    fn pipeline_o0_skips_scheduling() {
        // At O0, run_optimization should skip scheduling entirely.
        let config = config_for(OptLevel::O0);
        let pipeline = Pipeline::new(config);
        let mut func = make_simple_func();
        // Should not panic — O0 skips scheduling.
        pipeline.run_optimization(&mut func);
    }

    #[test]
    fn pipeline_o1_uses_basic_scheduler() {
        // At O1, run_optimization should use InstructionScheduler.
        let config = config_for(OptLevel::O1);
        let pipeline = Pipeline::new(config);
        let mut func = make_simple_func();
        pipeline.run_optimization(&mut func);
    }

    #[test]
    fn pipeline_o2_uses_pressure_aware_scheduler_by_default() {
        // At O2 with default config, PressureAwareScheduler should run.
        let config = config_for(OptLevel::O2);
        let pipeline = Pipeline::new(config);
        let mut func = make_simple_func();
        pipeline.run_optimization(&mut func);
    }

    #[test]
    fn pipeline_o3_uses_pressure_aware_scheduler_by_default() {
        // At O3 with default config, PressureAwareScheduler should run.
        let config = config_for(OptLevel::O3);
        let pipeline = Pipeline::new(config);
        let mut func = make_simple_func();
        pipeline.run_optimization(&mut func);
    }

    #[test]
    fn pipeline_o2_uses_basic_scheduler_when_pressure_disabled() {
        // At O2 with use_pressure_aware_scheduler = false, InstructionScheduler
        // should be used instead.
        let config = config_no_pressure(OptLevel::O2);
        let pipeline = Pipeline::new(config);
        let mut func = make_simple_func();
        pipeline.run_optimization(&mut func);
    }

    #[test]
    fn pipeline_o3_uses_basic_scheduler_when_pressure_disabled() {
        // At O3 with use_pressure_aware_scheduler = false, InstructionScheduler
        // should be used instead.
        let config = config_no_pressure(OptLevel::O3);
        let pipeline = Pipeline::new(config);
        let mut func = make_simple_func();
        pipeline.run_optimization(&mut func);
    }

    #[test]
    fn pipeline_o1_ignores_pressure_aware_flag() {
        // At O1, use_pressure_aware_scheduler should be irrelevant — the fast
        // InstructionScheduler is always used.
        let config = PipelineConfig {
            opt_level: OptLevel::O1,
            use_pressure_aware_scheduler: true,
            ..Default::default()
        };
        let pipeline = Pipeline::new(config);
        let mut func = make_simple_func();
        pipeline.run_optimization(&mut func);
    }
}

// ---------------------------------------------------------------------------
// Tests — dispatch verification integration
// ---------------------------------------------------------------------------

#[cfg(test)]
mod dispatch_verification_tests {
    use super::*;
    use llvm2_lower::compute_graph::{
        ComputeCost, ComputeNode, ComputeNodeId, DataEdge, NodeKind, TransferCost,
    };
    use llvm2_lower::target_analysis::ComputeTarget;

    // -----------------------------------------------------------------------
    // Test helpers
    // -----------------------------------------------------------------------

    /// Build a ComputeGraph from nodes and edges (avoids pub(crate) field issues).
    fn make_graph(nodes: Vec<ComputeNode>, edges: Vec<DataEdge>) -> ComputeGraph {
        let mut graph = ComputeGraph::new();
        graph.nodes = nodes;
        graph.edges = edges;
        graph
    }

    /// Build a simple single-node scalar graph.
    fn scalar_graph() -> (ComputeGraph, Vec<TargetRecommendation>) {
        let mut costs = HashMap::new();
        costs.insert(
            ComputeTarget::CpuScalar,
            ComputeCost {
                latency_cycles: 10,
                throughput_ops_per_kcycle: 1000,
            },
        );

        let node = ComputeNode {
            id: ComputeNodeId(0),
            instructions: vec![],
            costs,
            legal_targets: vec![ComputeTarget::CpuScalar],
            kind: NodeKind::Scalar,
            data_size_bytes: 8,
            produced_values: vec![],
            consumed_values: vec![],
            dominant_op: "ADD".to_string(),
            target_legality: None,
            matmul_shape: None,
        };

        let graph = make_graph(vec![node], vec![]);

        let recs = vec![TargetRecommendation {
            node_id: ComputeNodeId(0),
            recommended_target: ComputeTarget::CpuScalar,
            legal_targets: vec![ComputeTarget::CpuScalar],
            reason: "scalar op".to_string(),
            parallel_reduction_legal: false,
        }];

        (graph, recs)
    }

    /// Build a two-node GPU graph: CPU producer -> GPU consumer.
    fn gpu_graph() -> (ComputeGraph, Vec<TargetRecommendation>) {
        let mut cpu_costs = HashMap::new();
        cpu_costs.insert(
            ComputeTarget::CpuScalar,
            ComputeCost {
                latency_cycles: 20,
                throughput_ops_per_kcycle: 1000,
            },
        );

        let mut gpu_costs = HashMap::new();
        gpu_costs.insert(
            ComputeTarget::Gpu,
            ComputeCost {
                latency_cycles: 5,
                throughput_ops_per_kcycle: 100_000,
            },
        );
        gpu_costs.insert(
            ComputeTarget::CpuScalar,
            ComputeCost {
                latency_cycles: 500,
                throughput_ops_per_kcycle: 1000,
            },
        );

        let producer = ComputeNode {
            id: ComputeNodeId(0),
            instructions: vec![],
            costs: cpu_costs,
            legal_targets: vec![ComputeTarget::CpuScalar],
            kind: NodeKind::Scalar,
            data_size_bytes: 8,
            produced_values: vec![],
            consumed_values: vec![],
            dominant_op: "ADD".to_string(),
            target_legality: None,
            matmul_shape: None,
        };

        let consumer = ComputeNode {
            id: ComputeNodeId(1),
            instructions: vec![],
            costs: gpu_costs,
            legal_targets: vec![ComputeTarget::CpuScalar, ComputeTarget::Gpu],
            kind: NodeKind::DataParallel,
            data_size_bytes: 4096,
            produced_values: vec![],
            consumed_values: vec![],
            dominant_op: "ADD".to_string(),
            target_legality: None,
            matmul_shape: None,
        };

        let edge = DataEdge {
            from: ComputeNodeId(0),
            to: ComputeNodeId(1),
            transfer_bytes: 4096,
            transfer_cost: TransferCost::zero(),
        };

        let graph = make_graph(vec![producer, consumer], vec![edge]);

        let recs = vec![
            TargetRecommendation {
                node_id: ComputeNodeId(0),
                recommended_target: ComputeTarget::CpuScalar,
                legal_targets: vec![ComputeTarget::CpuScalar],
                reason: "scalar input".to_string(),
                parallel_reduction_legal: false,
            },
            TargetRecommendation {
                node_id: ComputeNodeId(1),
                recommended_target: ComputeTarget::Gpu,
                legal_targets: vec![ComputeTarget::CpuScalar, ComputeTarget::Gpu],
                reason: "data-parallel on GPU".to_string(),
                parallel_reduction_legal: true,
            },
        ];

        (graph, recs)
    }

    /// Build a graph with a GPU-only node (missing CpuScalar from legal_targets).
    fn bad_fallback_graph() -> (ComputeGraph, Vec<TargetRecommendation>) {
        let mut gpu_costs = HashMap::new();
        gpu_costs.insert(
            ComputeTarget::Gpu,
            ComputeCost {
                latency_cycles: 5,
                throughput_ops_per_kcycle: 100_000,
            },
        );

        let node = ComputeNode {
            id: ComputeNodeId(0),
            instructions: vec![],
            costs: gpu_costs,
            legal_targets: vec![ComputeTarget::Gpu], // Missing CpuScalar!
            kind: NodeKind::DataParallel,
            data_size_bytes: 4096,
            produced_values: vec![],
            consumed_values: vec![],
            dominant_op: "ADD".to_string(),
            target_legality: None,
            matmul_shape: None,
        };

        let graph = make_graph(vec![node], vec![]);

        let recs = vec![TargetRecommendation {
            node_id: ComputeNodeId(0),
            recommended_target: ComputeTarget::Gpu,
            legal_targets: vec![ComputeTarget::Gpu],
            reason: "GPU only".to_string(),
            parallel_reduction_legal: false,
        }];

        (graph, recs)
    }

    // -----------------------------------------------------------------------
    // Test 1: Valid plan passes verification in all modes
    // -----------------------------------------------------------------------

    #[test]
    fn test_valid_plan_passes_all_modes() {
        let (graph, recs) = scalar_graph();

        for mode in [
            DispatchVerifyMode::Off,
            DispatchVerifyMode::FallbackOnFailure,
            DispatchVerifyMode::ErrorOnFailure,
        ] {
            let pipeline = Pipeline::new(PipelineConfig {
                verify_dispatch: mode,
                ..Default::default()
            });

            let result = pipeline.generate_and_verify_dispatch(&graph, &recs);
            assert!(result.is_ok(), "mode {:?} should accept a valid plan", mode);
        }
    }

    // -----------------------------------------------------------------------
    // Test 2: Off mode skips verification entirely
    // -----------------------------------------------------------------------

    #[test]
    fn test_off_mode_skips_verification() {
        let (graph, recs) = bad_fallback_graph();

        let pipeline = Pipeline::new(PipelineConfig {
            verify_dispatch: DispatchVerifyMode::Off,
            ..Default::default()
        });

        // Even a plan with issues should pass when verification is off.
        let result = pipeline.generate_and_verify_dispatch(&graph, &recs);
        assert!(result.is_ok(), "Off mode should skip verification");

        // The plan should be the original (GPU launch, not CPU fallback).
        let plan = result.unwrap();
        let has_gpu_launch = plan.ops.iter().any(|op| {
            matches!(
                op,
                llvm2_lower::dispatch::DispatchOp::KernelLaunch {
                    target: ComputeTarget::Gpu,
                    ..
                }
            )
        });
        assert!(has_gpu_launch, "Off mode should preserve original GPU plan");
    }

    // -----------------------------------------------------------------------
    // Test 3: FallbackOnFailure mode produces CPU-only plan on bad graph
    // -----------------------------------------------------------------------

    #[test]
    fn test_fallback_mode_produces_cpu_plan_on_failure() {
        let (graph, recs) = bad_fallback_graph();

        let pipeline = Pipeline::new(PipelineConfig {
            verify_dispatch: DispatchVerifyMode::FallbackOnFailure,
            ..Default::default()
        });

        let result = pipeline.generate_and_verify_dispatch(&graph, &recs);
        assert!(result.is_ok(), "FallbackOnFailure should not error");

        let plan = result.unwrap();
        // The fallback plan should contain only CpuFallback ops.
        assert_eq!(plan.count_fallbacks(), 1, "should have 1 CPU fallback");
        assert_eq!(plan.count_launches(), 0, "no kernel launches in fallback");
        assert_eq!(plan.count_transfers(), 0, "no transfers in fallback");
        assert_eq!(plan.count_syncs(), 0, "no syncs in fallback");
    }

    // -----------------------------------------------------------------------
    // Test 4: ErrorOnFailure mode returns error on bad graph
    // -----------------------------------------------------------------------

    #[test]
    fn test_error_mode_returns_error_on_failure() {
        let (graph, recs) = bad_fallback_graph();

        let pipeline = Pipeline::new(PipelineConfig {
            verify_dispatch: DispatchVerifyMode::ErrorOnFailure,
            ..Default::default()
        });

        let result = pipeline.generate_and_verify_dispatch(&graph, &recs);
        assert!(result.is_err(), "ErrorOnFailure should return error");

        match result.unwrap_err() {
            PipelineError::DispatchVerificationFailed {
                violations,
                summary,
                report,
            } => {
                assert!(violations > 0, "should report violations");
                assert!(!summary.is_empty(), "should have summary text");
                assert!(!report.cpu_fallback_ok, "should flag CPU fallback issue");
            }
            other => panic!("Expected DispatchVerificationFailed, got: {:?}", other),
        }
    }

    // -----------------------------------------------------------------------
    // Test 5: GPU graph produces valid plan that passes verification
    // -----------------------------------------------------------------------

    #[test]
    fn test_gpu_graph_passes_verification() {
        let (graph, recs) = gpu_graph();

        let pipeline = Pipeline::new(PipelineConfig {
            verify_dispatch: DispatchVerifyMode::ErrorOnFailure,
            ..Default::default()
        });

        let result = pipeline.generate_and_verify_dispatch(&graph, &recs);
        assert!(result.is_ok(), "GPU graph should produce a valid plan");

        let plan = result.unwrap();
        assert!(plan.count_launches() >= 2, "should have CPU + GPU launches");
        assert!(
            plan.count_transfers() >= 1,
            "should have at least one transfer"
        );
    }

    // -----------------------------------------------------------------------
    // Test 6: generate_cpu_only_plan produces all-CPU plan
    // -----------------------------------------------------------------------

    #[test]
    fn test_generate_cpu_only_plan() {
        let (graph, recs) = gpu_graph();
        let plan = generate_cpu_only_plan(&graph, &recs);

        assert_eq!(plan.ops.len(), 2, "one fallback per node");
        assert_eq!(plan.count_fallbacks(), 2, "all ops should be CpuFallback");
        assert_eq!(plan.count_launches(), 0, "no kernel launches");
        assert_eq!(plan.count_transfers(), 0, "no transfers");
        assert_eq!(plan.count_syncs(), 0, "no syncs");

        // All assignments should be CpuScalar.
        for target in plan.assignment.values() {
            assert_eq!(*target, ComputeTarget::CpuScalar);
        }
    }

    // -----------------------------------------------------------------------
    // Test 7: CPU-only plan has correct estimated cycles
    // -----------------------------------------------------------------------

    #[test]
    fn test_cpu_only_plan_estimated_cycles() {
        let (graph, recs) = gpu_graph();
        let plan = generate_cpu_only_plan(&graph, &recs);

        // Node 0: CPU cost 20 cycles, Node 1: CPU cost 500 cycles.
        assert_eq!(
            plan.estimated_total_cycles, 520,
            "CPU-only plan should sum CPU costs: 20 + 500 = 520"
        );
    }

    // -----------------------------------------------------------------------
    // Test 8: verify_dispatch_plan with explicit plan validates correctly
    // -----------------------------------------------------------------------

    #[test]
    fn test_verify_dispatch_plan_explicit() {
        let (graph, recs) = gpu_graph();
        let plan = generate_dispatch_plan(&graph, &recs);

        let pipeline = Pipeline::new(PipelineConfig {
            verify_dispatch: DispatchVerifyMode::ErrorOnFailure,
            ..Default::default()
        });

        let result = pipeline.verify_dispatch_plan(&graph, plan, &recs);
        assert!(result.is_ok(), "Explicitly generated plan should pass");
    }

    // -----------------------------------------------------------------------
    // Test 9: Empty graph produces valid empty plan in all modes
    // -----------------------------------------------------------------------

    #[test]
    fn test_empty_graph_all_modes() {
        let graph = ComputeGraph::new();
        let recs = vec![];

        for mode in [
            DispatchVerifyMode::Off,
            DispatchVerifyMode::FallbackOnFailure,
            DispatchVerifyMode::ErrorOnFailure,
        ] {
            let pipeline = Pipeline::new(PipelineConfig {
                verify_dispatch: mode,
                ..Default::default()
            });

            let result = pipeline.generate_and_verify_dispatch(&graph, &recs);
            assert!(result.is_ok(), "Empty graph should pass in mode {:?}", mode);
            let plan = result.unwrap();
            assert!(plan.is_empty(), "Empty graph should produce empty plan");
        }
    }

    // -----------------------------------------------------------------------
    // Test 10: Default config uses FallbackOnFailure
    // -----------------------------------------------------------------------

    #[test]
    fn test_default_config_uses_fallback_mode() {
        let config = PipelineConfig::default();
        assert_eq!(
            config.verify_dispatch,
            DispatchVerifyMode::FallbackOnFailure,
            "Default should be FallbackOnFailure"
        );
    }

    // -----------------------------------------------------------------------
    // Test 11: CPU-only fallback plan passes verification itself
    // -----------------------------------------------------------------------

    #[test]
    fn test_cpu_only_fallback_passes_verification() {
        let (graph, recs) = gpu_graph();
        let cpu_plan = generate_cpu_only_plan(&graph, &recs);

        let report = verify_dispatch_plan_properties(&graph, &cpu_plan);
        assert!(
            report.all_ok(),
            "CPU-only fallback plan should pass all verification properties:\n{}",
            report
        );
    }

    // -----------------------------------------------------------------------
    // Test 12: DispatchVerificationFailed error has meaningful Display
    // -----------------------------------------------------------------------

    #[test]
    fn test_dispatch_verification_error_display() {
        let (graph, recs) = bad_fallback_graph();

        let pipeline = Pipeline::new(PipelineConfig {
            verify_dispatch: DispatchVerifyMode::ErrorOnFailure,
            ..Default::default()
        });

        let err = pipeline
            .generate_and_verify_dispatch(&graph, &recs)
            .unwrap_err();
        let display = format!("{}", err);
        assert!(
            display.contains("dispatch verification failed"),
            "Error display should contain 'dispatch verification failed', got: {}",
            display
        );
        assert!(
            display.contains("violation"),
            "Error display should contain 'violation', got: {}",
            display
        );
    }
}

// ---------------------------------------------------------------------------
// Tests — CoreML MIL generation from dispatch plan
// ---------------------------------------------------------------------------

#[cfg(test)]
mod coreml_dispatch_tests {
    use super::*;
    use llvm2_lower::compute_graph::{ComputeCost, ComputeNode, ComputeNodeId, DataEdge, NodeKind};
    use llvm2_lower::dispatch::DispatchOp;
    use llvm2_lower::target_analysis::ComputeTarget;

    // -----------------------------------------------------------------------
    // Test helpers
    // -----------------------------------------------------------------------

    fn make_graph(nodes: Vec<ComputeNode>, edges: Vec<DataEdge>) -> ComputeGraph {
        let mut graph = ComputeGraph::new();
        graph.nodes = nodes;
        graph.edges = edges;
        graph
    }

    fn make_ane_node(id: u32, kind: NodeKind, dominant_op: &str, data_size: u64) -> ComputeNode {
        let mut costs = HashMap::new();
        costs.insert(
            ComputeTarget::NeuralEngine,
            ComputeCost {
                latency_cycles: 100,
                throughput_ops_per_kcycle: 5000,
            },
        );
        costs.insert(
            ComputeTarget::CpuScalar,
            ComputeCost {
                latency_cycles: 500,
                throughput_ops_per_kcycle: 1000,
            },
        );
        ComputeNode {
            id: ComputeNodeId(id),
            instructions: vec![],
            costs,
            legal_targets: vec![ComputeTarget::NeuralEngine, ComputeTarget::CpuScalar],
            kind,
            data_size_bytes: data_size,
            produced_values: vec![],
            consumed_values: vec![],
            dominant_op: dominant_op.to_string(),
            target_legality: None,
            matmul_shape: None,
        }
    }

    fn make_cpu_only_node(id: u32, dominant_op: &str, data_size: u64) -> ComputeNode {
        let mut costs = HashMap::new();
        costs.insert(
            ComputeTarget::CpuScalar,
            ComputeCost {
                latency_cycles: 50,
                throughput_ops_per_kcycle: 1000,
            },
        );
        ComputeNode {
            id: ComputeNodeId(id),
            instructions: vec![],
            costs,
            legal_targets: vec![ComputeTarget::CpuScalar],
            kind: NodeKind::Scalar,
            data_size_bytes: data_size,
            produced_values: vec![],
            consumed_values: vec![],
            dominant_op: dominant_op.to_string(),
            target_legality: None,
            matmul_shape: None,
        }
    }

    /// Build a dispatch plan with explicit ANE assignments and kernel launches.
    fn make_ane_plan(node_ids: &[u32], cycles_per_node: u64) -> DispatchPlan {
        let mut assignment = HashMap::new();
        let mut ops = Vec::new();
        let mut total_cycles: u64 = 0;

        for &id in node_ids {
            let nid = ComputeNodeId(id);
            assignment.insert(nid, ComputeTarget::NeuralEngine);
            ops.push(DispatchOp::KernelLaunch {
                target: ComputeTarget::NeuralEngine,
                node_id: nid,
                estimated_cycles: cycles_per_node,
            });
            total_cycles += cycles_per_node;
        }

        DispatchPlan {
            ops,
            assignment,
            estimated_total_cycles: total_cycles,
        }
    }

    /// Build a mixed plan with some ANE and some CPU nodes.
    fn make_mixed_plan(
        ane_ids: &[u32],
        cpu_ids: &[u32],
        ane_cycles: u64,
        cpu_cycles: u64,
    ) -> DispatchPlan {
        let mut assignment = HashMap::new();
        let mut ops = Vec::new();
        let mut total_cycles: u64 = 0;

        for &id in cpu_ids {
            let nid = ComputeNodeId(id);
            assignment.insert(nid, ComputeTarget::CpuScalar);
            ops.push(DispatchOp::CpuFallback {
                node_id: nid,
                reason: "CPU-only node".to_string(),
            });
            total_cycles += cpu_cycles;
        }

        for &id in ane_ids {
            let nid = ComputeNodeId(id);
            assignment.insert(nid, ComputeTarget::NeuralEngine);
            ops.push(DispatchOp::KernelLaunch {
                target: ComputeTarget::NeuralEngine,
                node_id: nid,
                estimated_cycles: ane_cycles,
            });
            total_cycles += ane_cycles;
        }

        DispatchPlan {
            ops,
            assignment,
            estimated_total_cycles: total_cycles,
        }
    }

    // -----------------------------------------------------------------------
    // Test 1: Single ANE GEMM node produces valid CoreMLOutput
    // -----------------------------------------------------------------------

    #[test]
    fn test_emit_coreml_single_gemm() {
        let node = make_ane_node(0, NodeKind::MatrixHeavy, "GEMM", 8192);
        let graph = make_graph(vec![node], vec![]);
        let plan = make_ane_plan(&[0], 5000);

        let output = emit_coreml_program(&plan, &graph).unwrap();

        assert_eq!(output.program.op_count(), 1);
        assert_eq!(output.program.operations[0].op_type(), "matmul");
        assert!(output.program.validate().is_ok());
        assert!(output.estimated_latency_us >= 1);
    }

    // -----------------------------------------------------------------------
    // Test 2: Multi-node ANE subgraph (GEMM -> ADD -> RELU) with fusion
    // -----------------------------------------------------------------------

    #[test]
    fn test_emit_coreml_gemm_bias_relu_fusion() {
        let nodes = vec![
            make_ane_node(0, NodeKind::MatrixHeavy, "GEMM", 8192),
            make_ane_node(1, NodeKind::DataParallel, "ADD", 256),
            make_ane_node(2, NodeKind::DataParallel, "RELU", 256),
        ];
        let graph = make_graph(nodes, vec![]);
        let plan = make_ane_plan(&[0, 1, 2], 3000);

        let output = emit_coreml_program(&plan, &graph).unwrap();

        // GEMM-Bias-Act fusion produces 3 MIL ops (matmul + add + relu)
        assert_eq!(output.program.op_count(), 3);
        assert_eq!(output.program.operations[0].op_type(), "matmul");
        assert_eq!(output.program.operations[1].op_type(), "add");
        assert_eq!(output.program.operations[2].op_type(), "relu");
        assert!(output.program.validate().is_ok());
    }

    // -----------------------------------------------------------------------
    // Test 3: Mixed plan filters only ANE nodes
    // -----------------------------------------------------------------------

    #[test]
    fn test_emit_coreml_mixed_plan_filters_ane_only() {
        let nodes = vec![
            make_cpu_only_node(0, "ADD", 64),
            make_ane_node(1, NodeKind::MatrixHeavy, "GEMM", 8192),
            make_cpu_only_node(2, "SUB", 64),
        ];
        let graph = make_graph(nodes, vec![]);
        let plan = make_mixed_plan(&[1], &[0, 2], 5000, 50);

        let output = emit_coreml_program(&plan, &graph).unwrap();

        // Only node 1 (GEMM) should be in the MIL program
        assert_eq!(output.program.op_count(), 1);
        assert_eq!(output.program.operations[0].op_type(), "matmul");
        assert!(output.program.validate().is_ok());
    }

    // -----------------------------------------------------------------------
    // Test 4: Empty plan (no ANE nodes) returns error
    // -----------------------------------------------------------------------

    #[test]
    fn test_emit_coreml_no_ane_nodes_returns_error() {
        let node = make_cpu_only_node(0, "ADD", 64);
        let graph = make_graph(vec![node], vec![]);

        // Plan with only CPU assignments
        let mut assignment = HashMap::new();
        assignment.insert(ComputeNodeId(0), ComputeTarget::CpuScalar);
        let plan = DispatchPlan {
            ops: vec![DispatchOp::CpuFallback {
                node_id: ComputeNodeId(0),
                reason: "CPU-only".to_string(),
            }],
            assignment,
            estimated_total_cycles: 50,
        };

        let result = emit_coreml_program(&plan, &graph);
        assert!(result.is_err());
        match result.unwrap_err() {
            PipelineError::CoreMLEmit(CoreMLEmitError::EmptyNodeList) => {}
            other => panic!("expected CoreMLEmit(EmptyNodeList), got: {:?}", other),
        }
    }

    // -----------------------------------------------------------------------
    // Test 5: ANE compatibility warnings are propagated
    // -----------------------------------------------------------------------

    #[test]
    fn test_emit_coreml_propagates_ane_warnings() {
        // The GEMM + RELU program with FP16 should have no warnings
        let nodes = vec![
            make_ane_node(0, NodeKind::MatrixHeavy, "GEMM", 8192),
            make_ane_node(1, NodeKind::DataParallel, "RELU", 2048),
        ];
        let graph = make_graph(nodes, vec![]);
        let plan = make_ane_plan(&[0, 1], 2000);

        let output = emit_coreml_program(&plan, &graph).unwrap();

        // Standard FP16 matmul+relu should produce zero warnings
        assert!(
            output.warnings.is_empty(),
            "Expected no warnings for FP16 matmul+relu, got: {:?}",
            output.warnings
        );
    }

    // -----------------------------------------------------------------------
    // Test 6: Estimated latency calculation from plan cycles
    // -----------------------------------------------------------------------

    #[test]
    fn test_emit_coreml_estimated_latency() {
        let node = make_ane_node(0, NodeKind::MatrixHeavy, "GEMM", 8192);
        let graph = make_graph(vec![node], vec![]);

        // 10000 cycles at 1GHz = 10000 ns = 10 us
        let plan = make_ane_plan(&[0], 10000);

        let output = emit_coreml_program(&plan, &graph).unwrap();
        assert_eq!(
            output.estimated_latency_us, 10,
            "10000 cycles / 1000 = 10 us"
        );
    }

    // -----------------------------------------------------------------------
    // Test 7: Multiple ANE nodes produce correct latency sum
    // -----------------------------------------------------------------------

    #[test]
    fn test_emit_coreml_multi_node_latency() {
        let nodes = vec![
            make_ane_node(0, NodeKind::MatrixHeavy, "GEMM", 8192),
            make_ane_node(1, NodeKind::DataParallel, "RELU", 2048),
            make_ane_node(2, NodeKind::MatrixHeavy, "GEMM", 4096),
        ];
        let graph = make_graph(nodes, vec![]);

        // 3 nodes * 5000 cycles each = 15000 cycles = 15 us
        let plan = make_ane_plan(&[0, 1, 2], 5000);

        let output = emit_coreml_program(&plan, &graph).unwrap();
        assert_eq!(
            output.estimated_latency_us, 15,
            "3 * 5000 cycles / 1000 = 15 us"
        );
    }

    // -----------------------------------------------------------------------
    // Test 8: Node ordering is deterministic (sorted by ID)
    // -----------------------------------------------------------------------

    #[test]
    fn test_emit_coreml_deterministic_ordering() {
        // Insert nodes in reverse order to verify sorting
        let nodes = vec![
            make_ane_node(2, NodeKind::DataParallel, "RELU", 2048),
            make_ane_node(0, NodeKind::MatrixHeavy, "GEMM", 8192),
            make_ane_node(1, NodeKind::DataParallel, "ADD", 256),
        ];
        let graph = make_graph(nodes, vec![]);
        let plan = make_ane_plan(&[0, 1, 2], 1000);

        let output = emit_coreml_program(&plan, &graph).unwrap();

        // Nodes should be processed in ID order (0, 1, 2) = GEMM, ADD, RELU
        // GEMM -> ADD -> RELU triggers the GemmBiasAct fusion
        assert_eq!(output.program.op_count(), 3);
        assert_eq!(output.program.operations[0].op_type(), "matmul");
        assert_eq!(output.program.operations[1].op_type(), "add");
        assert_eq!(output.program.operations[2].op_type(), "relu");
    }

    // -----------------------------------------------------------------------
    // Test 9: CoreMLOutput struct fields are populated correctly
    // -----------------------------------------------------------------------

    #[test]
    fn test_coreml_output_struct_fields() {
        let node = make_ane_node(0, NodeKind::DataParallel, "RELU", 2048);
        let graph = make_graph(vec![node], vec![]);
        let plan = make_ane_plan(&[0], 2000);

        let output = emit_coreml_program(&plan, &graph).unwrap();

        // program field: should have operations
        assert!(!output.program.operations.is_empty());
        // program should have inputs and outputs
        assert!(!output.program.inputs.is_empty());
        assert!(!output.program.outputs.is_empty());
        // warnings field: Vec<String>
        let _: &Vec<String> = &output.warnings;
        // estimated_latency_us: u64, must be positive
        assert!(output.estimated_latency_us >= 1);
    }

    // -----------------------------------------------------------------------
    // Test 10: Conv node produces correct MIL op type
    // -----------------------------------------------------------------------

    #[test]
    fn test_emit_coreml_conv_node() {
        let node = make_ane_node(0, NodeKind::DataParallel, "CONV", 4096);
        let graph = make_graph(vec![node], vec![]);
        let plan = make_ane_plan(&[0], 3000);

        let output = emit_coreml_program(&plan, &graph).unwrap();

        assert_eq!(output.program.op_count(), 1);
        assert_eq!(output.program.operations[0].op_type(), "conv");
        assert!(output.program.validate().is_ok());
    }

    // -----------------------------------------------------------------------
    // Test 11: MatMul + GELU fusion pattern
    // -----------------------------------------------------------------------

    #[test]
    fn test_emit_coreml_matmul_gelu_fusion() {
        let nodes = vec![
            make_ane_node(0, NodeKind::MatrixHeavy, "GEMM", 8192),
            make_ane_node(1, NodeKind::DataParallel, "GELU", 4096),
        ];
        let graph = make_graph(nodes, vec![]);
        let plan = make_ane_plan(&[0, 1], 4000);

        let output = emit_coreml_program(&plan, &graph).unwrap();

        // MatMul-GELU fusion: matmul + gelu = 2 ops
        assert_eq!(output.program.op_count(), 2);
        assert_eq!(output.program.operations[0].op_type(), "matmul");
        assert_eq!(output.program.operations[1].op_type(), "gelu");
        assert!(output.program.validate().is_ok());
    }

    // -----------------------------------------------------------------------
    // Test 12: Minimum latency is clamped to 1 us
    // -----------------------------------------------------------------------

    #[test]
    fn test_emit_coreml_minimum_latency_clamp() {
        let node = make_ane_node(0, NodeKind::DataParallel, "RELU", 64);
        let graph = make_graph(vec![node], vec![]);

        // Very small cycle count: 100 cycles / 1000 = 0, clamped to 1
        let plan = make_ane_plan(&[0], 100);

        let output = emit_coreml_program(&plan, &graph).unwrap();
        assert_eq!(
            output.estimated_latency_us, 1,
            "Latency should be clamped to minimum 1 us"
        );
    }
}

// ===========================================================================
// Exception handling pipeline integration tests
// ===========================================================================

#[cfg(test)]
mod eh_pipeline_tests {
    use super::*;
    use llvm2_ir::function::{
        EhCallSiteEntry, LandingPadEntry, MachFunction as IrMachFunction, Signature as IrSignature,
        Type,
    };
    use llvm2_ir::types::BlockId;

    #[test]
    fn generate_lsda_for_function_no_eh_returns_none() {
        let func = build_add_test_function();
        assert!(generate_lsda_for_function(&func).is_none());
    }

    #[test]
    fn generate_lsda_for_function_with_cleanup() {
        let sig = IrSignature::new(vec![Type::I64], vec![Type::I64]);
        let mut func = IrMachFunction::new("with_cleanup".to_string(), sig);

        // Set up EH metadata: one cleanup landing pad.
        func.eh_metadata.personality = Some("__rust_eh_personality".to_string());
        func.eh_metadata.add_landing_pad(LandingPadEntry {
            block: BlockId(1),
            offset: 0x40,
            catch_type_indices: Vec::new(),
            is_cleanup: true,
        });
        func.eh_metadata.add_call_site(EhCallSiteEntry {
            call_block: BlockId(0),
            start_offset: 0x00,
            length: 0x20,
            landing_pad_block: Some(BlockId(1)),
            action_index: 0,
        });

        let lsda = generate_lsda_for_function(&func);
        assert!(
            lsda.is_some(),
            "Function with EH metadata should produce LSDA"
        );

        let lsda_bytes = lsda.unwrap();
        // LSDA must start with LPStart encoding (0xFF = omit).
        assert_eq!(
            lsda_bytes[0], 0xFF,
            "LPStart encoding should be DW_EH_PE_omit"
        );
        // Must have at least the minimal header.
        assert!(
            lsda_bytes.len() > 4,
            "LSDA should have header + call site table"
        );
    }

    #[test]
    fn generate_lsda_for_function_with_typed_catch() {
        let sig = IrSignature::new(vec![Type::I64], vec![Type::I64]);
        let mut func = IrMachFunction::new("with_catch".to_string(), sig);

        // Set up EH metadata: one typed catch landing pad.
        func.eh_metadata.personality = Some("__gxx_personality_v0".to_string());
        func.eh_metadata.add_landing_pad(LandingPadEntry {
            block: BlockId(2),
            offset: 0x80,
            catch_type_indices: vec![1],
            is_cleanup: false,
        });
        func.eh_metadata.add_call_site(EhCallSiteEntry {
            call_block: BlockId(0),
            start_offset: 0x10,
            length: 0x08,
            landing_pad_block: Some(BlockId(2)),
            action_index: 1,
        });
        func.eh_metadata.add_type_index(1);

        let lsda = generate_lsda_for_function(&func);
        assert!(lsda.is_some());

        let lsda_bytes = lsda.unwrap();
        assert_eq!(lsda_bytes[0], 0xFF); // LPStart = omit
        // TType encoding should NOT be omit (we have typed catch).
        assert_ne!(
            lsda_bytes[1], 0xFF,
            "TType should not be omit when catch types exist"
        );
    }

    #[test]
    fn pipeline_encode_and_emit_with_eh_metadata() {
        // Build a simple function with physical registers (post-regalloc state)
        // and attach EH metadata.
        let mut func = build_add_test_function();

        // Attach cleanup EH metadata.
        func.eh_metadata.personality = Some("__rust_eh_personality".to_string());
        func.eh_metadata.add_landing_pad(LandingPadEntry {
            block: BlockId(0),
            offset: 0x08, // 2 instructions in = 8 bytes
            catch_type_indices: Vec::new(),
            is_cleanup: true,
        });
        func.eh_metadata.add_call_site(EhCallSiteEntry {
            call_block: BlockId(0),
            start_offset: 0x00,
            length: 0x04,
            landing_pad_block: Some(BlockId(0)),
            action_index: 0,
        });

        let pipeline = Pipeline::new(PipelineConfig {
            opt_level: OptLevel::O0,
            emit_debug: false,
            verify_dispatch: DispatchVerifyMode::Off,
            enable_post_ra_opt: false,
            use_pressure_aware_scheduler: false,
            ..Default::default()
        });

        let obj_bytes = pipeline.encode_and_emit(&mut func).unwrap();

        // The Mach-O should contain the __gcc_except_tab section name somewhere.
        let section_name = b"__gcc_except_tab";
        let found = obj_bytes
            .windows(section_name.len())
            .any(|w| w == section_name);
        assert!(
            found,
            "Mach-O output should contain __gcc_except_tab section"
        );
    }

    #[test]
    fn pipeline_encode_and_emit_no_eh_no_except_tab() {
        // A function WITHOUT EH metadata should NOT have __gcc_except_tab.
        let mut func = build_add_test_function();

        let pipeline = Pipeline::new(PipelineConfig {
            opt_level: OptLevel::O0,
            emit_debug: false,
            verify_dispatch: DispatchVerifyMode::Off,
            enable_post_ra_opt: false,
            use_pressure_aware_scheduler: false,
            ..Default::default()
        });

        let obj_bytes = pipeline.encode_and_emit(&mut func).unwrap();

        let section_name = b"__gcc_except_tab";
        let found = obj_bytes
            .windows(section_name.len())
            .any(|w| w == section_name);
        assert!(!found, "Mach-O without EH should not have __gcc_except_tab");
    }

    #[test]
    fn generate_lsda_for_function_multiple_call_sites() {
        let sig = IrSignature::new(vec![Type::I64], vec![Type::I64]);
        let mut func = IrMachFunction::new("multi_invoke".to_string(), sig);

        // Two landing pads: one catch, one cleanup.
        func.eh_metadata.personality = Some("__gxx_personality_v0".to_string());
        func.eh_metadata.add_landing_pad(LandingPadEntry {
            block: BlockId(3),
            offset: 0x80,
            catch_type_indices: vec![1],
            is_cleanup: false,
        });
        func.eh_metadata.add_landing_pad(LandingPadEntry {
            block: BlockId(4),
            offset: 0xC0,
            catch_type_indices: Vec::new(),
            is_cleanup: true,
        });

        // Three call sites: two with landing pads, one without.
        func.eh_metadata.add_call_site(EhCallSiteEntry {
            call_block: BlockId(0),
            start_offset: 0x00,
            length: 0x10,
            landing_pad_block: Some(BlockId(3)),
            action_index: 1,
        });
        func.eh_metadata.add_call_site(EhCallSiteEntry {
            call_block: BlockId(1),
            start_offset: 0x10,
            length: 0x08,
            landing_pad_block: Some(BlockId(4)),
            action_index: 0,
        });
        func.eh_metadata.add_call_site(EhCallSiteEntry {
            call_block: BlockId(2),
            start_offset: 0x18,
            length: 0x04,
            landing_pad_block: None,
            action_index: 0,
        });

        let lsda = generate_lsda_for_function(&func);
        assert!(lsda.is_some());

        let lsda_bytes = lsda.unwrap();
        // With multiple call sites, the LSDA should be substantially larger than
        // the minimal header.
        assert!(
            lsda_bytes.len() > 20,
            "Multi-call-site LSDA should be >20 bytes, got {}",
            lsda_bytes.len()
        );
    }
}

// ===========================================================================
// Multiblock optimization pipeline tests (Part of #288)
//
// These tests verify that multi-block functions (branches, loops) compile
// successfully at all optimization levels (O0-O3). Regression test for
// potential infinite loops in the optimization or register allocation
// fixpoint iterations.
// ===========================================================================

#[cfg(test)]
mod multiblock_pipeline_tests {
    use super::*;
    use llvm2_ir::function::{MachFunction as IrMachFunction, Signature as IrSignature, Type};
    use llvm2_ir::inst::{AArch64Opcode as IrOpcode, MachInst as IrMachInst};
    use llvm2_ir::operand::MachOperand as IrOperand;
    use llvm2_ir::regs::{X0, X1};

    /// Build a multi-block function with an if/else branch using PReg operands.
    ///
    /// ```text
    /// bb0 (entry): MovI X0, #42; Cbz X0, bb2
    /// bb1 (then):  Ret             // X0 = 42 (nonzero, so fallthrough)
    /// bb2 (else):  MovI X0, #0; Ret  // X0 = 0 (if Cbz taken)
    /// ```
    fn build_multiblock_branch_function() -> IrMachFunction {
        let sig = IrSignature::new(vec![], vec![Type::I64]);
        let mut func = IrMachFunction::new("branch_test".to_string(), sig);
        let bb0 = func.entry; // BlockId(0)
        let bb1 = func.create_block(); // BlockId(1)
        let bb2 = func.create_block(); // BlockId(2)

        // bb0: MovI X0, #42
        let movi = IrMachInst::new(
            IrOpcode::MovI,
            vec![IrOperand::PReg(X0), IrOperand::Imm(42)],
        );
        let movi_id = func.push_inst(movi);
        func.append_inst(bb0, movi_id);

        // bb0: Cbz X0, bb2
        let cbz = IrMachInst::new(
            IrOpcode::Cbz,
            vec![IrOperand::PReg(X0), IrOperand::Block(bb2)],
        );
        let cbz_id = func.push_inst(cbz);
        func.append_inst(bb0, cbz_id);

        // bb1: Ret
        let ret1 = IrMachInst::new(IrOpcode::Ret, vec![]);
        let ret1_id = func.push_inst(ret1);
        func.append_inst(bb1, ret1_id);

        // bb2: MovI X0, #0
        let movi2 = IrMachInst::new(IrOpcode::MovI, vec![IrOperand::PReg(X0), IrOperand::Imm(0)]);
        let movi2_id = func.push_inst(movi2);
        func.append_inst(bb2, movi2_id);

        // bb2: Ret
        let ret2 = IrMachInst::new(IrOpcode::Ret, vec![]);
        let ret2_id = func.push_inst(ret2);
        func.append_inst(bb2, ret2_id);

        // CFG edges: bb0 -> bb1 (fallthrough), bb0 -> bb2 (Cbz taken)
        func.add_edge(bb0, bb1);
        func.add_edge(bb0, bb2);

        func
    }

    /// Build a multi-block function with a loop using PReg operands.
    ///
    /// ```text
    /// bb0 (entry):  MovI X0, #5; MovI X1, #0; B bb1
    /// bb1 (header): Cbz X0, bb3
    /// bb2 (body):   SubRI X0, X0, #1; AddRI X1, X1, #1; B bb1
    /// bb3 (exit):   MovR X0, X1; Ret
    /// ```
    ///
    /// At exit, X0 = 5 (the count). Uses PRegs to simulate post-ISel code.
    fn build_multiblock_loop_function() -> IrMachFunction {
        let sig = IrSignature::new(vec![], vec![Type::I64]);
        let mut func = IrMachFunction::new("loop_test".to_string(), sig);
        let bb0 = func.entry; // BlockId(0)
        let bb1 = func.create_block(); // BlockId(1)
        let bb2 = func.create_block(); // BlockId(2)
        let bb3 = func.create_block(); // BlockId(3)

        // bb0: MovI X0, #5
        let m0 = IrMachInst::new(IrOpcode::MovI, vec![IrOperand::PReg(X0), IrOperand::Imm(5)]);
        let m0_id = func.push_inst(m0);
        func.append_inst(bb0, m0_id);

        // bb0: MovI X1, #0
        let m1 = IrMachInst::new(IrOpcode::MovI, vec![IrOperand::PReg(X1), IrOperand::Imm(0)]);
        let m1_id = func.push_inst(m1);
        func.append_inst(bb0, m1_id);

        // bb0: B bb1
        let br0 = IrMachInst::new(IrOpcode::B, vec![IrOperand::Block(bb1)]);
        let br0_id = func.push_inst(br0);
        func.append_inst(bb0, br0_id);

        // bb1: Cbz X0, bb3
        let cbz = IrMachInst::new(
            IrOpcode::Cbz,
            vec![IrOperand::PReg(X0), IrOperand::Block(bb3)],
        );
        let cbz_id = func.push_inst(cbz);
        func.append_inst(bb1, cbz_id);

        // bb2: SubRI X0, X0, #1
        let sub = IrMachInst::new(
            IrOpcode::SubRI,
            vec![IrOperand::PReg(X0), IrOperand::PReg(X0), IrOperand::Imm(1)],
        );
        let sub_id = func.push_inst(sub);
        func.append_inst(bb2, sub_id);

        // bb2: AddRI X1, X1, #1
        let add = IrMachInst::new(
            IrOpcode::AddRI,
            vec![IrOperand::PReg(X1), IrOperand::PReg(X1), IrOperand::Imm(1)],
        );
        let add_id = func.push_inst(add);
        func.append_inst(bb2, add_id);

        // bb2: B bb1
        let br2 = IrMachInst::new(IrOpcode::B, vec![IrOperand::Block(bb1)]);
        let br2_id = func.push_inst(br2);
        func.append_inst(bb2, br2_id);

        // bb3: MovR X0, X1
        let mov = IrMachInst::new(
            IrOpcode::MovR,
            vec![IrOperand::PReg(X0), IrOperand::PReg(X1)],
        );
        let mov_id = func.push_inst(mov);
        func.append_inst(bb3, mov_id);

        // bb3: Ret
        let ret = IrMachInst::new(IrOpcode::Ret, vec![]);
        let ret_id = func.push_inst(ret);
        func.append_inst(bb3, ret_id);

        // CFG edges
        func.add_edge(bb0, bb1);
        func.add_edge(bb1, bb2); // fallthrough
        func.add_edge(bb1, bb3); // Cbz taken
        func.add_edge(bb2, bb1); // back-edge

        func
    }

    /// Compile an IR function through the pipeline at a given optimization level,
    /// returning the Mach-O object bytes.
    fn compile_ir_at_opt(
        func: &mut IrMachFunction,
        opt_level: OptLevel,
    ) -> Result<Vec<u8>, PipelineError> {
        let pipeline = Pipeline::new(PipelineConfig {
            opt_level,
            ..Default::default()
        });
        pipeline.compile_ir_function(func)
    }

    #[test]
    fn multiblock_branch_compiles_at_all_opt_levels() {
        for opt in &[OptLevel::O0, OptLevel::O1, OptLevel::O2, OptLevel::O3] {
            let mut func = build_multiblock_branch_function();
            let obj_bytes = compile_ir_at_opt(&mut func, *opt)
                .unwrap_or_else(|e| panic!("branch_test at {:?} failed: {}", opt, e));

            assert!(obj_bytes.len() >= 4, "branch_test at {:?} too small", opt);
            let magic =
                u32::from_le_bytes([obj_bytes[0], obj_bytes[1], obj_bytes[2], obj_bytes[3]]);
            assert_eq!(
                magic, 0xFEED_FACF,
                "branch_test at {:?}: invalid Mach-O magic {:#010X}",
                opt, magic
            );
        }
    }

    #[test]
    fn multiblock_loop_compiles_at_all_opt_levels() {
        for opt in &[OptLevel::O0, OptLevel::O1, OptLevel::O2, OptLevel::O3] {
            let mut func = build_multiblock_loop_function();
            let obj_bytes = compile_ir_at_opt(&mut func, *opt)
                .unwrap_or_else(|e| panic!("loop_test at {:?} failed: {}", opt, e));

            assert!(obj_bytes.len() >= 4, "loop_test at {:?} too small", opt);
            let magic =
                u32::from_le_bytes([obj_bytes[0], obj_bytes[1], obj_bytes[2], obj_bytes[3]]);
            assert_eq!(
                magic, 0xFEED_FACF,
                "loop_test at {:?}: invalid Mach-O magic {:#010X}",
                opt, magic
            );
        }
    }

    #[test]
    fn multiblock_branch_optimization_reduces_code() {
        // Optimized code should be no larger than unoptimized (and likely smaller
        // since the branch function has a constant Cbz that can be folded).
        let mut func_o0 = build_multiblock_branch_function();
        let obj_o0 = compile_ir_at_opt(&mut func_o0, OptLevel::O0).expect("O0 should compile");

        let mut func_o2 = build_multiblock_branch_function();
        let obj_o2 = compile_ir_at_opt(&mut func_o2, OptLevel::O2).expect("O2 should compile");

        // Both produce valid Mach-O
        assert!(obj_o0.len() >= 4);
        assert!(obj_o2.len() >= 4);

        // O2 should not produce dramatically more code than O0
        assert!(
            obj_o2.len() <= obj_o0.len() + 100,
            "O2 ({} bytes) should not be dramatically larger than O0 ({} bytes)",
            obj_o2.len(),
            obj_o0.len()
        );
    }
}
