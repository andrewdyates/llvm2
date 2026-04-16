# tRust-LLVM2 Bridge: Integration Design

**Date:** 2026-04-16
**Author:** Andrew Yates <ayates@dropbox.com>
**Status:** Draft
**Part of:** #259, #266

---

## Executive Summary

This document designs the integration bridge between tRust (508K LOC, 34 crates) and LLVM2 (120K LOC, 6 crates). The bridge must translate tRust's `VerifiableFunction` (place-based MIR) into a format LLVM2 can compile, propagate verification results through compilation, and return proof certificates proving the binary matches the source semantics.

Two integration paths are specified: (1) a **library link** for in-process compilation where tRust depends on LLVM2 as a Cargo git dependency, and (2) a **JSON wire format + CLI** for out-of-process compilation. Both paths use the same tMIR serialization format.

**Key finding from the t* stack audit (2026-04-16):** tRust does NOT produce tMIR today. Its IR is `trust-types::VerifiableFunction`, a place-based MIR representation with sidecar proof dispositions. A bridge crate (`trust-llvm2-bridge`) is required to translate VerifiableFunction into the tMIR format LLVM2 consumes. The 924-line LaTeX design at `tRust/designs/2026-04-14-trust-llvm2-integration.tex` specifies this bridge in detail.

---

## 1. API Contract Mapping

### 1.1 tRust's CodegenBackend Trait

Source: `~/tRust/crates/trust-driver/src/codegen.rs`

```rust
pub(crate) trait CodegenBackend {
    fn name(&self) -> &str;
    fn is_available(&self) -> bool;
    fn lower_mir(&self, function_name: &str, mir_body: &MirBody) -> Result<LoweredFunction, CodegenError>;
    fn emit_object(&self, lowered: &[LoweredFunction], output_path: &Path) -> Result<CodegenArtifact, CodegenError>;
    fn verify_lowering(&self, function_name: &str, mir_body: &MirBody, lowered: &LoweredFunction) -> Result<VerificationResult, CodegenError>;
}
```

### 1.2 LLVM2's Public Compiler API

Source: `crates/llvm2-codegen/src/compiler.rs`

```rust
pub struct Compiler { config: CompilerConfig }

impl Compiler {
    pub fn new(config: CompilerConfig) -> Self;
    pub fn compile(&self, module: &tmir_func::Module) -> Result<CompilationResult, CompileError>;
    pub fn compile_ir_function(&self, ir_func: &mut MachFunction) -> Result<CompilationResult, CompileError>;
}
```

### 1.3 Contract Mapping

| tRust CodegenBackend | LLVM2 API | Notes |
|----------------------|-----------|-------|
| `lower_mir(name, mir_body)` | `llvm2_lower::translate_module(module)` | Bridge converts MirBody to tmir_func::Module, then LLVM2's adapter translates to LIR |
| `emit_object(lowered, path)` | `Compiler::compile(module)` returns `CompilationResult { object_code: Vec<u8> }` | tRust writes `object_code` to `output_path` |
| `verify_lowering(name, mir, lowered)` | `Compiler::compile()` with `emit_proofs: true` returns `proofs: Option<Vec<ProofCertificate>>` | Currently stub; requires z4 wiring |
| `is_available()` | Feature detection: check `llvm2_codegen` crate compiles and z4 is present | Library mode: always true if linked. CLI mode: check binary exists. |
| `name()` | Returns `"llvm2"` | Already matches |

### 1.4 Bridged Llvm2Backend Implementation

The `Llvm2Backend` in tRust currently holds `tool_path: Option<PathBuf>` and `available: bool`. The bridged implementation will operate in two modes:

```rust
// In trust-llvm2-bridge crate (lives in tRust repo)
pub struct BridgedLlvm2Backend {
    /// In-process compiler (library mode).
    compiler: Option<llvm2_codegen::Compiler>,
    /// CLI binary path (CLI mode, fallback).
    cli_path: Option<PathBuf>,
}
```

**Library mode** (preferred): `trust-llvm2-bridge` depends on `llvm2-codegen` as a git dependency. The `Compiler` struct is used directly in-process. No serialization overhead for the hot path.

**CLI mode** (fallback): tRust serializes the module to tMIR JSON, invokes `llvm2-cli`, and parses the output. Used when the library dependency is not available or for debugging.

---

## 2. Serialization Format

### 2.1 Current State

LLVM2's stub crate `tmir-func` already has a complete JSON reader/writer at `stubs/tmir-func/src/reader.rs`:
- `read_module_from_json(path)` / `read_module_from_str(json)` / `read_module_from_reader(reader)`
- `write_module_to_json(module, path)` / `write_module_to_string(module)`
- Structural validation (non-empty names, entry block existence)
- Round-trip tested

All tMIR stub types (`Ty`, `Instr`, `Block`, `Function`, `Module`, `TmirProof`) derive `serde::Serialize` and `serde::Deserialize`.

### 2.2 Format Decision: JSON for Wire, In-Process for Library

| Mode | Format | When |
|------|--------|------|
| Library link (in-process) | No serialization; direct Rust struct passing | tRust depends on `llvm2-codegen` via git dep |
| CLI invocation | JSON (serde_json) | tRust invokes `llvm2-cli` binary |
| Debug/test fixtures | JSON files on disk | Manual testing, CI, reproduction |

**Why JSON, not protobuf or bincode:**
1. LLVM2's stubs already have full serde JSON support (reader.rs, 237 LOC with tests).
2. Human-readable for debugging the bridge during development.
3. No additional dependencies. Both tRust and LLVM2 already depend on `serde` + `serde_json`.
4. Performance is not a concern for the wire format because the library mode (no serialization) is the primary path. JSON is only for CLI fallback and test fixtures.
5. If performance becomes a concern later, switching to `bincode` or `rmp-serde` (MessagePack) requires changing exactly one line (`serde_json::to_string` -> `bincode::serialize`).

### 2.3 tMIR JSON Schema

The wire format is defined by the serde serialization of `tmir_func::Module`:

```json
{
  "name": "my_crate",
  "functions": [
    {
      "id": {"0": 0},
      "name": "add_i32",
      "ty": { "params": [{"Primitive": {"Int": {"width": "I32", "signed": true}}}, ...], "returns": [...] },
      "entry": {"0": 0},
      "blocks": [
        {
          "id": {"0": 0},
          "params": [[ {"0": 0}, {"Primitive": {"Int": {"width": "I32", "signed": true}}} ]],
          "body": [
            {
              "instr": {"BinOp": {"op": "Add", "ty": ..., "lhs": {"0": 0}, "rhs": {"0": 1}, "result": {"0": 2}}},
              "metadata": null,
              "proofs": []
            }
          ]
        }
      ],
      "proofs": ["Pure"]
    }
  ],
  "structs": []
}
```

### 2.4 Bridge Translation: VerifiableFunction to tmir_func::Module

This is the critical path (issue #266). The bridge in tRust must:

1. **Type mapping:** `trust_types::Ty` -> `tmir_types::Ty` (direct for scalars; restricted for Stage 1)
2. **Place-to-SSA conversion:** MIR's `Statement::Assign{place, rvalue}` -> block-parameter SSA (Cytron algorithm, restricted to scalar locals in Stage 1)
3. **Instruction mapping:** `trust_types::Rvalue::{BinaryOp, UnaryOp, ...}` -> `tmir_instrs::Instr::{BinOp, UnOp, ...}`
4. **Terminator mapping:** `Terminator::Return/Goto/SwitchInt` -> `Instr::Return/Jump/BranchIf`
5. **Proof annotation propagation:** `TrustDisposition` -> `tmir_types::TmirProof` via the ObligationMap (see Section 3)

**Stage 1 restrictions (scalar-only subset):**
- Functions with only local scalar variables (no references, no projections)
- No calls (leaf functions only)
- No drop glue
- Types: Bool, Int{8,16,32,64,128}, Float{32,64}

---

## 3. Verification Result Alignment

### 3.1 tRust's VerificationResult

Source: `~/tRust/crates/trust-types/src/result.rs`

```rust
#[non_exhaustive]
pub enum VerificationResult {
    Proved { solver: String, time_ms: u64, strength: ProofStrength,
             proof_certificate: Option<Vec<u8>>, solver_warnings: Option<Vec<String>> },
    Failed { solver: String, time_ms: u64, counterexample: Option<Counterexample> },
    Unknown { solver: String, time_ms: u64, reason: String },
    Timeout { solver: String, timeout_ms: u64 },
}

pub enum ProofStrength {
    SmtUnsat,         // z4 returned UNSAT
    Bounded(u64),     // BMC checked to depth k
    Inductive,        // invariant found
    Deductive,        // proved under preconditions
    Constructive,     // lean5 proof term
    Certified,        // lean5 kernel verified
}
```

### 3.2 LLVM2's ProofCertificate

Source: `crates/llvm2-codegen/src/compiler.rs`

```rust
pub struct ProofCertificate {
    pub rule_name: String,
    pub verified: bool,
}
```

### 3.3 Alignment Plan

LLVM2's `ProofCertificate` is a placeholder. It must be enriched to align with tRust's `VerificationResult`. The recommended approach:

**Option A (shared type, preferred):** Both tRust and LLVM2 depend on a shared `verification-types` crate (or use `trust-types` directly). LLVM2 already depends on a local stub of tmir-types which has `TmirProof`; extending it to include `VerificationResult` is natural.

**Option B (compatible but independent):** LLVM2 defines its own `LoweringVerificationResult` that is `From<trust_types::VerificationResult>` and vice versa. This avoids coupling LLVM2's release cycle to trust-types.

**Recommended: Option A for in-process, Option B for CLI.**

For the library link path, the bridge crate in tRust already depends on both crate ecosystems and can use tRust's `VerificationResult` directly. For the CLI path, LLVM2 outputs a JSON verification report that the tRust CLI wrapper deserializes into its own `VerificationResult`.

**LLVM2 enriched result type (proposed):**

```rust
/// Enriched verification result for lowering proofs.
///
/// Replaces the placeholder ProofCertificate with full verification metadata
/// compatible with trust-types::VerificationResult.
pub enum LoweringVerification {
    /// Lowering proved correct by SMT solver.
    Proved {
        rule_name: String,
        solver: String,
        time_ms: u64,
        strength: LoweringProofStrength,
        certificate: Option<Vec<u8>>,
    },
    /// Lowering found to be incorrect (counterexample found).
    Failed {
        rule_name: String,
        solver: String,
        time_ms: u64,
        counterexample: Option<String>,
    },
    /// Verification inconclusive.
    Unknown {
        rule_name: String,
        solver: String,
        time_ms: u64,
        reason: String,
    },
    /// Solver timed out.
    Timeout {
        rule_name: String,
        solver: String,
        timeout_ms: u64,
    },
}

pub enum LoweringProofStrength {
    /// z4 SMT solver returned UNSAT (strongest for bitvector proofs).
    SmtUnsat,
    /// Exhaustive evaluation for small bit-widths (mock evaluator).
    Exhaustive { width_bits: u32 },
    /// Random sampling (weak, statistical confidence only).
    Sampled { num_samples: u64 },
    /// CEGIS synthesis verified the rewrite rule.
    CegisSynthesis,
}
```

### 3.4 Conversion Functions

```rust
// In trust-llvm2-bridge (tRust repo)
impl From<llvm2_codegen::LoweringVerification> for trust_types::VerificationResult {
    fn from(lv: LoweringVerification) -> Self {
        match lv {
            LoweringVerification::Proved { solver, time_ms, strength, certificate, .. } => {
                VerificationResult::Proved {
                    solver,
                    time_ms,
                    strength: match strength {
                        LoweringProofStrength::SmtUnsat => ProofStrength::SmtUnsat,
                        LoweringProofStrength::CegisSynthesis => ProofStrength::Deductive,
                        _ => ProofStrength::Bounded(0), // degraded
                    },
                    proof_certificate: certificate,
                    solver_warnings: None,
                }
            }
            LoweringVerification::Failed { solver, time_ms, counterexample, .. } => {
                VerificationResult::Failed {
                    solver,
                    time_ms,
                    counterexample: counterexample.map(|s| /* parse */),
                }
            }
            LoweringVerification::Unknown { solver, time_ms, reason, .. } => {
                VerificationResult::Unknown { solver, time_ms, reason }
            }
            LoweringVerification::Timeout { solver, timeout_ms, .. } => {
                VerificationResult::Timeout { solver, timeout_ms }
            }
        }
    }
}
```

---

## 4. Integration Architecture

### 4.1 Architecture Options Analysis

| Option | Latency | Complexity | Isolation | Recommended |
|--------|---------|------------|-----------|-------------|
| **A. Library link** (shared crate) | Lowest | Medium | None | Primary |
| **B. CLI + JSON** (subprocess) | Highest | Low | Full | Fallback |
| **C. IPC (Unix socket/shared memory)** | Medium | High | Medium | Deferred |
| **D. Shared crate (tmir types only)** | Low | Low | Partial | Stepping stone |

### 4.2 Recommended: Library Link (A) with CLI Fallback (B)

```
                    +----- Library Mode (primary) -----+
                    |                                    |
tRust               |  trust-llvm2-bridge (tRust repo)   |  LLVM2
                    |                                    |
VerifiableFunction  |  lower_to_tmir() ──> tmir::Module  |  Compiler::compile()
      +             |  ──────> llvm2_codegen::Compiler   |  ──> CompilationResult
ProofDispositions   |  <── LoweringVerification[]        |
                    |                                    |
                    +------------------------------------+

                    +---- CLI Mode (fallback/debug) -----+
                    |                                     |
tRust               |  trust-llvm2-bridge                 |  llvm2-cli
                    |                                     |
VerifiableFunction  |  lower_to_tmir() -> JSON file       |  < stdin: JSON
      +             |  exec("llvm2-cli", [json_path])     |  > stdout: .o bytes
ProofDispositions   |  parse(stdout) -> CodegenArtifact   |  > stderr: diag JSON
                    |                                     |
                    +-------------------------------------+
```

### 4.3 Dependency Graph

**Library mode:**
```
tRust/Cargo.toml:
  [dependencies]
  llvm2-codegen = { git = "https://github.com/ayates_dbx/LLVM2", features = ["library"] }

LLVM2/Cargo.toml (already exists):
  [workspace.dependencies]
  tmir-types = { path = "stubs/tmir-types" }    # will become git dep on tMIR
  tmir-func = { path = "stubs/tmir-func" }
```

**CLI mode:**
```
tRust/Cargo.toml:
  # No LLVM2 dependency; invokes llvm2-cli binary
  [dependencies]
  serde_json = "1.0"                             # for JSON serialization
  tmir-func = { path = "../tMIR/crates/tmir-func" }  # for Module struct + writer
```

### 4.4 Crate Boundary Design

The bridge consists of three logical components, physically split across two repos:

**In LLVM2 repo:**
1. `llvm2-codegen` (already exists) -- enriched with `LoweringVerification` type and `fn verify_lowering()` API
2. `llvm2-cli` (already exists) -- enhanced to accept `--verify` flag and output verification JSON

**In tRust repo:**
3. `trust-llvm2-bridge` (new crate) -- translates VerifiableFunction to tmir_func::Module, invokes LLVM2, maps results back

### 4.5 Thread Safety and Compilation Model

LLVM2's `Compiler` is stateless (no mutable shared state between compilations). Each call to `compile()` creates a fresh `Pipeline`. This means:

- **Parallel function compilation:** tRust can spawn multiple `Compiler::compile()` calls on different modules concurrently.
- **No global state:** No mutex, no global allocator state, no mutable statics.
- **One Compiler per thread:** Each thread creates its own `Compiler` instance.

This matches tRust's codegen model where functions can be compiled independently.

---

## 5. Phase Plan for Incremental Integration

### Phase 1: Scaffold (LLVM2 side)

**Goal:** LLVM2 exposes a stable library API that tRust can depend on.

**Tasks:**
- [ ] Enrich `ProofCertificate` -> `LoweringVerification` with solver/time/strength fields (LLVM2 #259)
- [ ] Add `pub fn verify_function(module: &tmir_func::Module, func_name: &str) -> Vec<LoweringVerification>` to Compiler
- [ ] Ensure `llvm2-codegen` compiles as a library (no binary entrypoint in lib.rs)
- [ ] Export `CompilationResult`, `CompilerConfig`, `LoweringVerification` from crate root
- [ ] Add JSON output mode to `llvm2-cli`: `--output-format json` emits metrics + verification results

**Deliverable:** `llvm2-codegen` v0.1.0 with stable API contract.

### Phase 2: Bridge Crate (tRust side)

**Goal:** tRust can translate a scalar-only VerifiableFunction to tMIR JSON.

**Tasks:**
- [ ] Create `trust-llvm2-bridge` crate in tRust repo (issue #266)
- [ ] Implement `lower_to_tmir(func: &VerifiableFunction) -> Result<tmir_func::Module, BridgeError>` for scalar subset
- [ ] Implement type mapping: `trust_types::Ty` -> `tmir_types::Ty` (scalars only)
- [ ] Implement SSA construction for scalar locals (simplified Cytron, no places)
- [ ] Implement instruction mapping: BinOp, UnOp, CmpOp, constants
- [ ] Implement terminator mapping: Return, Goto, SwitchInt->BranchIf
- [ ] Implement proof annotation mapping: TrustDisposition -> TmirProof
- [ ] Integration test: compile `fn add(a: i32, b: i32) -> i32 { a + b }` through the full path

**Deliverable:** `trust-llvm2-bridge` compiles one function end-to-end.

### Phase 3: Wire Up Llvm2Backend (tRust side)

**Goal:** tRust's `Llvm2Backend` uses the bridge crate for real compilation.

**Tasks:**
- [ ] Replace stub implementations in `Llvm2Backend` with bridge calls
- [ ] Library mode: `lower_mir()` calls `lower_to_tmir()` then `Compiler::compile()`
- [ ] CLI mode: `lower_mir()` calls `lower_to_tmir()`, writes JSON, invokes `llvm2-cli`
- [ ] `emit_object()` writes `CompilationResult.object_code` to output path
- [ ] `verify_lowering()` calls `Compiler::verify_function()`, maps to `VerificationResult`
- [ ] Set `available: true` when backend is detected

**Deliverable:** `Llvm2Backend::is_available() == true` with real compilation.

### Phase 4: Expand Type Support

**Goal:** Handle references, structs, enums (beyond scalars).

**Tasks:**
- [ ] Pointer/reference types: `&T`, `*const T` -> `tmir_types::Ty::Ref`
- [ ] Struct types with field layout
- [ ] Enum types with discriminant layout
- [ ] Array types
- [ ] Tuple types
- [ ] Expand SSA construction to handle Place projections (field access, deref)

**Deliverable:** Non-trivial Rust functions compile through the bridge.

### Phase 5: Full Verification Pipeline

**Goal:** End-to-end proof certificates from source to binary.

**Tasks:**
- [ ] LLVM2's z4 integration produces real proof certificates (not stubs)
- [ ] Certificate chain: tRust proofs + bridge obligation map + LLVM2 lowering proofs
- [ ] Translation validation at each compilation phase
- [ ] Non-local property propagation (ownership, lifetimes -- requires LIR enrichment)
- [ ] Certificate serialization format (JSON or CBOR)

**Deliverable:** `cargo +trust build --verified` produces binary + certificate bundle.

---

## 6. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| tMIR stubs diverge from real tMIR | High | Medium | Upstream stubs to tMIR repo; converge representations |
| Place-to-SSA conversion complexity | Medium | High | Stage 1 restricts to scalar locals; full Cytron deferred to Phase 4 |
| z4 API instability | Low | Medium | Pin z4 version in Cargo.toml; LLVM2 already handles 28+ API mismatches |
| Verification result mismatch | Low | Low | Both use z4 as solver; ProofStrength enum is extensible (`#[non_exhaustive]`) |
| Performance of JSON wire format | Low | Low | Library mode avoids serialization; JSON only for CLI fallback |
| Circular dependency (tRust -> LLVM2 -> tmir stubs ~ tRust types) | Medium | High | Bridge crate in tRust depends on LLVM2 git dep; LLVM2 depends on tmir stubs (not trust-types) |

---

## 7. Open Questions

1. **Where should tmir-types canonical definitions live?** Currently LLVM2's stubs (1,714 LOC) are more complete than the real tMIR repo (844 LOC). Proposal: upstream LLVM2's stubs to the tMIR repo and switch LLVM2's Cargo.toml to git deps on tMIR.

2. **Should trust-llvm2-bridge depend on llvm2-codegen or only on tmir-func?** If it depends on llvm2-codegen, it gets library mode. If it depends only on tmir-func, it is restricted to CLI mode but has zero LLVM2 build-time coupling. Recommendation: feature-gated dependency (`[features] library = ["llvm2-codegen"]`).

3. **How to handle incremental compilation?** tRust's `rustc` incremental model recompiles only changed functions. LLVM2's `Compiler::compile()` takes a whole `Module`. Options: (a) compile each function as a separate single-function module, (b) add `Compiler::compile_function()` API.

4. **Proof certificate format?** tRust's `proof_certificate: Option<Vec<u8>>` is opaque bytes. LLVM2's z4 bridge produces LRAT certificates. Should we standardize on LRAT or define a structured envelope format?

---

## References

- tRust codegen backend: `~/tRust/crates/trust-driver/src/codegen.rs` (542 LOC)
- tRust VerificationResult: `~/tRust/crates/trust-types/src/result.rs`
- tRust integration design: `~/tRust/designs/2026-04-14-trust-llvm2-integration.tex` (924 lines)
- LLVM2 Compiler API: `crates/llvm2-codegen/src/compiler.rs`
- LLVM2 tMIR adapter: `crates/llvm2-lower/src/adapter.rs`
- LLVM2 JSON reader: `stubs/tmir-func/src/reader.rs`
- t* stack audit: `designs/2026-04-16-t-star-integration-audit.md`
- Real tMIR integration design: `designs/2026-04-15-real-tmir-integration.md`
