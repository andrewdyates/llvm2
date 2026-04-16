# Proof Certificate Chain: tRust Source to LLVM2 Binary

**Date:** 2026-04-16
**Author:** Andrew Yates <ayates@dropbox.com>
**Status:** Draft
**Part of:** #269

---

## Executive Summary

This document designs the proof certificate chain that connects tRust's source-level
verification (function contracts, ownership proofs, safety properties) to LLVM2's
machine-level verification (semantic equivalence of lowered instructions). The goal
is an unbroken chain of machine-checkable certificates from `cargo +trust build` to
the final binary, proving that the binary faithfully implements the source-level
specification.

**Key insight:** tRust's `trust-proof-cert` crate (23K LOC) already has a mature
certificate infrastructure including hash-linked chains, proof composition, DAG-based
verification, and Ed25519 signing. LLVM2 does not need to invent a new format -- it
needs to produce certificates compatible with trust-proof-cert's `ProofCertificate`
type, add a new `ChainStepType` for machine lowering, and implement a
`LoweringCertificateGenerator` that produces these certificates from its existing
verification results.

---

## 1. Certificate Architecture

### 1.1 The Three Verification Phases

```
Phase 1: Source Verification (tRust)
  Input:  Rust source + #[requires], #[ensures] annotations
  Output: ProofCertificate per obligation (trust-proof-cert)
  Proves: Function f satisfies its contract for all inputs

Phase 2: Lowering Verification (LLVM2)
  Input:  tMIR function + AArch64 machine instructions
  Output: LoweringCertificate per lowering rule (this design)
  Proves: Machine code for f is semantically equivalent to tMIR for f

Phase 3: End-to-End Composition
  Input:  Phase 1 certificate + Phase 2 certificate
  Output: ComposedProof (trust-proof-cert composition)
  Proves: Machine code for f satisfies the source-level contract for f
```

### 1.2 Composition Theorem

The end-to-end correctness theorem is:

```
Given:
  (a) tRust proves: forall inputs, not UB(inputs) => f_source(inputs) refines spec(f)
  (b) LLVM2 proves: forall inputs, f_machine(inputs) = f_tmir(inputs)
  (c) Bridge proves: f_tmir = translate(f_source)  [structural, not SMT]

Then by transitivity:
  forall inputs, not UB(inputs) => f_machine(inputs) refines spec(f)
```

This is the standard verified compilation theorem from CompCert (Leroy 2006),
adapted for the two-phase verification architecture.

**Critical assumption (c):** The trust-llvm2-bridge translation from
VerifiableFunction to tmir_func::Module must be semantics-preserving. In Phase 1
(scalar-only), this is straightforward because the translation is a direct
instruction mapping. For Phase 4+ (references, structs), this becomes a separate
verification obligation (see Section 5.3).

---

## 2. tRust Certificate Infrastructure (Existing)

### 2.1 ProofCertificate (trust-proof-cert/cert.rs)

tRust's `ProofCertificate` is the canonical certificate type. It contains:

| Field | Type | Purpose |
|-------|------|---------|
| `id` | `CertificateId` | SHA-256 of (function, timestamp) |
| `function` | `String` | Fully-qualified function name |
| `function_hash` | `FunctionHash` | SHA-256 of function body |
| `vc_hash` | `[u8; 32]` | SHA-256 of verification condition |
| `vc_snapshot` | `VcSnapshot` | Kind + formula JSON + source location |
| `solver` | `SolverInfo` | Solver name, version, time, ProofStrength |
| `proof_steps` | `Vec<ProofStep>` | Solver reasoning trace |
| `witness` | `Option<Vec<u8>>` | Existential proof witnesses |
| `chain` | `CertificateChain` | Hash-linked step sequence |
| `proof_trace` | `Vec<u8>` | Raw solver proof bytes |
| `timestamp` | `String` | ISO 8601 |
| `status` | `CertificationStatus` | Trusted or Certified (lean5) |
| `signature` | `Option<CertificateSignature>` | Ed25519 signature |

Source: `~/tRust/crates/trust-proof-cert/src/cert.rs:133-164`

### 2.2 CertificateChain (trust-proof-cert/chain.rs)

The chain is an ordered sequence of `ChainStep`, each with:
- `step_type`: `VcGeneration | SolverProof | Lean5Certification`
- `input_hash` / `output_hash`: SHA-256 hashes linking steps
- `tool`, `tool_version`, `time_ms`, `timestamp`

Integrity check: `steps[i].output_hash == steps[i+1].input_hash`

### 2.3 Composition (trust-proof-cert/composition/)

The composition module provides:
- `compose_proofs(a, b)`: Merge two ProofCertificates into a ComposedProof
- `check_composability(a, b)`: Verify two certificates can be composed
- `transitive_closure(certs)`: Close over call-graph dependencies
- `ProofComposition` (DAG): Full dependency graph with cycle detection
- `ComposedProof`: Combined strength (weakest constituent), combined status

Source: `~/tRust/crates/trust-proof-cert/src/composition/`

### 2.4 What tRust Does NOT Have

- **No lowering certificate type.** ChainStepType has `VcGeneration`, `SolverProof`,
  `Lean5Certification` -- but no step type for "machine code lowering verified."
- **No cross-system composition.** Composition assumes all certificates use
  trust-types' `ProofStrength` and `VerificationCondition`. LLVM2 uses `SmtExpr`
  and its own `VerificationResult`.
- **No binary hash.** ProofCertificate has `function_hash` (source body hash) but
  no field for the hash of the emitted machine code bytes.

---

## 3. LLVM2 Certificate Infrastructure (Proposed)

### 3.1 Design Principles

1. **Produce trust-proof-cert-compatible certificates.** LLVM2 must emit
   certificates that tRust's composition module can consume. This means using
   the same types or providing lossless conversion.

2. **Reuse, don't reinvent.** trust-proof-cert has 23K LOC of certificate
   infrastructure. LLVM2 should generate certificates that plug into it, not
   build a parallel infrastructure.

3. **Hash everything.** Every input (tMIR module, compiler config) and output
   (machine code bytes) gets SHA-256 hashed into the chain.

4. **Granularity: per-function.** One certificate per compiled function, containing
   all lowering rule proofs for that function. This matches tRust's per-function
   `ProofCertificate` and enables function-level composition.

### 3.2 New ChainStepType: MachineLowering

LLVM2 certificates require a new `ChainStepType` variant. This should be proposed
upstream to trust-proof-cert:

```rust
// Proposed addition to trust-proof-cert/chain.rs
#[non_exhaustive]
pub enum ChainStepType {
    VcGeneration,
    SolverProof,
    Lean5Certification,
    /// NEW: Machine code lowering verified by LLVM2.
    /// Input: tMIR function hash. Output: machine code hash.
    MachineLowering,
}
```

Until upstream adds this, LLVM2 can use a local extension type that converts
to/from JSON with a string tag.

### 3.3 LoweringCertificate: LLVM2's Output

```rust
/// A lowering proof certificate for a single function.
///
/// Contains all the evidence that the machine code for a function
/// is semantically equivalent to its tMIR input. Compatible with
/// trust-proof-cert's ProofCertificate via conversion.
pub struct LoweringCertificate {
    /// Function name (matches tMIR function name).
    pub function: String,

    /// SHA-256 of the tMIR function (serialized to canonical JSON).
    pub tmir_hash: [u8; 32],

    /// SHA-256 of the emitted machine code bytes for this function.
    pub machine_code_hash: [u8; 32],

    /// Per-rule proof results.
    /// Each entry proves one lowering rule (e.g., "Iadd_I32 -> ADDWrr").
    pub rule_proofs: Vec<RuleProof>,

    /// Overall verification strength for this function.
    /// This is the weakest strength across all rule_proofs.
    pub overall_strength: LoweringProofStrength,

    /// Solver information.
    pub solver: String,
    pub solver_version: String,
    pub total_time_ms: u64,

    /// ISO 8601 timestamp.
    pub timestamp: String,

    /// Compiler configuration used (opt level, target).
    pub compiler_config_hash: [u8; 32],
}

/// Proof result for a single lowering rule within a function.
pub struct RuleProof {
    /// Rule name (e.g., "Iadd_I32 -> ADDWrr", "Isub_I64 -> SUBXrr").
    pub rule_name: String,

    /// Translation validation check kind (aligned with trust-transval).
    pub check_kind: TransvalCheckKind,

    /// Verification outcome.
    pub result: LoweringVerification,

    /// SHA-256 of the SMT formula (for replay/audit).
    pub formula_hash: [u8; 32],
}

/// Verification result for a lowering rule.
///
/// Structurally aligned with trust-types::VerificationResult.
pub enum LoweringVerification {
    /// Lowering proved correct.
    Proved {
        strength: LoweringProofStrength,
        time_ms: u64,
        /// Raw z4 UNSAT proof bytes (when available).
        proof_trace: Option<Vec<u8>>,
    },
    /// Counterexample found.
    Failed {
        time_ms: u64,
        counterexample: String,
    },
    /// Solver could not determine.
    Unknown {
        time_ms: u64,
        reason: String,
    },
    /// Solver timed out.
    Timeout {
        timeout_ms: u64,
    },
}

/// Proof strength levels for lowering verification.
///
/// Ordered by increasing assurance. Maps to trust-types ProofStrength
/// via the conversion functions in Section 3.5.
pub enum LoweringProofStrength {
    /// Statistical sampling (weakest -- not a proof).
    /// N random inputs tested with no counterexample.
    Sampled { num_samples: u64 },

    /// Exhaustive evaluation for small bit-widths.
    /// All 2^(width * num_inputs) combinations tested.
    Exhaustive { width_bits: u32 },

    /// CEGIS synthesis verified the rewrite rule.
    /// Counter-example guided, iterative, but terminates with proof.
    CegisSynthesis,

    /// SMT solver (z4) returned UNSAT.
    /// Strongest: universal proof for all inputs.
    SmtUnsat,
}
```

### 3.4 Certificate Generation

The `LoweringCertificateGenerator` sits between the compiler pipeline and the
certificate output:

```rust
/// Generates lowering certificates during compilation.
///
/// Wired into Compiler::compile() when emit_proofs is true.
/// For each function:
///   1. Hash the tMIR input (canonical JSON serialization)
///   2. Run lowering + verification (existing ProofObligation pipeline)
///   3. Hash the machine code output
///   4. Bundle rule proofs into a LoweringCertificate
pub struct LoweringCertificateGenerator {
    config: CertificateConfig,
}

pub struct CertificateConfig {
    /// Minimum proof strength required for certificate emission.
    /// Functions with any rule below this threshold get no certificate.
    pub min_strength: LoweringProofStrength,

    /// Whether to include raw proof traces (z4 UNSAT proofs).
    /// These can be large; omit for space-constrained deployments.
    pub include_proof_traces: bool,
}
```

Integration point in `Compiler::compile()`:

```rust
// In Compiler::compile(), after pipeline completes:
if self.config.emit_proofs {
    let cert_gen = LoweringCertificateGenerator::new(cert_config);
    for (lir_func, proof_ctx) in &lir_functions {
        let cert = cert_gen.generate(
            &lir_func.name,
            &tmir_module,      // for tMIR hash
            &machine_code,     // for machine code hash
            &proof_ctx,        // existing proof results
        )?;
        certificates.push(cert);
    }
}
```

### 3.5 Conversion to trust-proof-cert Types

LLVM2's `LoweringCertificate` must convert to trust-proof-cert's `ProofCertificate`
for composition. The conversion maps are:

**LoweringProofStrength -> trust_types::ProofStrength:**

| LLVM2 | trust-types Reasoning | trust-types Assurance |
|-------|----------------------|----------------------|
| `Sampled { n }` | `BoundedModelCheck { depth: n }` | `BoundedSound { depth: n }` |
| `Exhaustive { w }` | `ExhaustiveFinite(2^w)` | `Sound` |
| `CegisSynthesis` | `Deductive` | `Sound` |
| `SmtUnsat` | `Smt` | `Sound` |

Note: `Sampled` is an imperfect mapping -- BMC explores states exhaustively to
a depth, while sampling tests random inputs. The mapping preserves the key
semantic property (not a complete proof) but the reasoning kinds differ. A more
precise mapping would add a `StatisticalSampling` variant to trust-types'
`ReasoningKind`; this should be proposed upstream.

**LoweringCertificate -> ProofCertificate:**

```rust
impl LoweringCertificate {
    /// Convert to a trust-proof-cert ProofCertificate.
    ///
    /// The resulting certificate can be composed with tRust source-level
    /// certificates using trust-proof-cert's composition module.
    pub fn to_proof_certificate(&self) -> ProofCertificate {
        let vc_snapshot = VcSnapshot {
            kind: "lowering_equivalence".to_string(),
            formula_json: self.serialize_formula_summary(),
            location: None,  // machine-level, no source location
        };

        let solver_info = SolverInfo {
            name: self.solver.clone(),
            version: self.solver_version.clone(),
            time_ms: self.total_time_ms,
            strength: self.overall_strength.to_proof_strength(),
            evidence: None,
        };

        let mut chain = CertificateChain::new();

        // Step 1: tMIR hash -> lowering verification
        chain.push(ChainStep {
            step_type: ChainStepType::MachineLowering, // new variant
            tool: "llvm2".to_string(),
            tool_version: env!("CARGO_PKG_VERSION").to_string(),
            input_hash: hex::encode(self.tmir_hash),
            output_hash: hex::encode(self.machine_code_hash),
            time_ms: self.total_time_ms,
            timestamp: self.timestamp.clone(),
        });

        // Step 2: z4 proof (when SmtUnsat)
        if matches!(self.overall_strength, LoweringProofStrength::SmtUnsat) {
            chain.push(ChainStep {
                step_type: ChainStepType::SolverProof,
                tool: "z4".to_string(),
                tool_version: "latest".to_string(),
                input_hash: hex::encode(self.machine_code_hash),
                output_hash: hex::encode(self.compute_proof_hash()),
                time_ms: self.total_time_ms,
                timestamp: self.timestamp.clone(),
            });
        }

        ProofCertificate {
            id: CertificateId::generate(&self.function, &self.timestamp),
            function: self.function.clone(),
            function_hash: FunctionHash::from_bytes(&self.tmir_hash),
            vc_hash: vc_snapshot.vc_hash(),
            vc_snapshot,
            solver: solver_info,
            proof_steps: self.extract_proof_steps(),
            witness: None,
            chain,
            proof_trace: self.collect_proof_traces(),
            timestamp: self.timestamp.clone(),
            status: CertificationStatus::Trusted,
            version: CERT_FORMAT_VERSION,
            signature: None,
        }
    }
}
```

---

## 4. End-to-End Certificate Chain

### 4.1 Chain Structure

For a function `fn add(a: i32, b: i32) -> i32 { a + b }`:

```
Certificate 1 (tRust, source-level):
  function: "crate::add"
  vc_kind: "postcondition"
  proves: add(a, b) returns a + b for all i32 a, b (where no overflow)
  chain: [VcGeneration] -> [SolverProof(z4)]
  status: Trusted
  strength: { reasoning: Smt, assurance: Sound }

Certificate 2 (LLVM2, lowering):
  function: "crate::add"
  vc_kind: "lowering_equivalence"
  proves: ADDWrr(a, b) = tmir::BinOp::Add(a, b) for all 32-bit inputs
  chain: [MachineLowering(llvm2)] -> [SolverProof(z4)]
  status: Trusted
  strength: { reasoning: Smt, assurance: Sound }

Composed Certificate (trust-proof-cert composition):
  constituent_ids: [cert1.id, cert2.id]
  functions: ["crate::add"]
  proves: the AArch64 binary for add satisfies the source postcondition
  combined_strength: { reasoning: Smt, assurance: Sound }  (weakest of constituents)
  combined_status: Trusted  (both are Trusted)
```

### 4.2 Composition Workflow

```
1. tRust compiles with verification:
   cargo +trust build --verified
       -> VerificationResult::Proved for each obligation
       -> ProofCertificate per obligation (stored in target/trust-certs/)

2. LLVM2 compiles tMIR to machine code:
   Compiler::compile(module) with emit_proofs: true
       -> LoweringCertificate per function
       -> Convert to ProofCertificate via to_proof_certificate()

3. Bridge composes the certificates:
   trust_proof_cert::composition::compose_proofs(source_cert, lowering_cert)
       -> ComposedProof if composable
       -> Error if incompatible (different functions, contradictory assumptions)

4. Final bundle:
   All ComposedProofs written to binary's companion .proof file
   or embedded in a custom Mach-O section (__LLVM2,__proofs)
```

### 4.3 Composability Requirements

For two certificates to compose:

1. **Same function:** Both must certify the same function name.
2. **Hash linkage:** The lowering certificate's `tmir_hash` must match the
   source certificate's `function_hash` (or a translation thereof).
3. **No contradictory assumptions:** Source cert's postcondition must be
   compatible with lowering cert's preconditions.
4. **Strength compatibility:** Both must have a defined ProofStrength.

Requirement (2) is the subtle one. The source cert hashes the MIR body,
while the lowering cert hashes the tMIR representation. These are different
representations of the same function. The bridge must produce a
"translation certificate" that links MIR_hash to tMIR_hash (see Section 5.3).

### 4.4 Certificate Storage

**Option A (companion file):** Write certificates to `<binary>.proof` alongside
the binary. JSON or CBOR format, one file per compilation unit.

**Option B (embedded section):** Embed in a custom Mach-O section
`__LLVM2,__proofs`. This keeps certificates co-located with the binary, survives
file moves, and can be stripped with `strip -R __LLVM2 binary`.

**Recommended:** Option A for development (human-readable JSON), Option B for
production (embedded, tamper-resistant when combined with signing).

---

## 5. Gap Analysis

### 5.1 What LLVM2 Must Build

| Component | Status | Priority | Effort |
|-----------|--------|----------|--------|
| `LoweringCertificate` struct | New | High | Small |
| `LoweringCertificateGenerator` | New | High | Medium |
| `LoweringProofStrength` enum | Proposed in bridge design | High | Small |
| `to_proof_certificate()` conversion | New | High | Medium |
| Wire into `Compiler::compile()` | Modify existing | High | Small |
| `RuleProof` per-instruction evidence | New | Medium | Medium |
| Machine code hash computation | New | Medium | Small |
| tMIR canonical serialization + hash | New | Medium | Small |
| Certificate JSON output in CLI | New | Low | Small |
| Mach-O `__LLVM2,__proofs` section | New | Low | Medium |

### 5.2 What trust-proof-cert Must Add (Upstream)

| Component | Status | Notes |
|-----------|--------|-------|
| `ChainStepType::MachineLowering` | Proposed | Single enum variant addition |
| `binary_hash` field on ProofCertificate | Proposed | Optional field for machine code hash |
| `StatisticalSampling` reasoning kind | Nice-to-have | For LLVM2's `Sampled` strength |

These are small upstream changes. LLVM2 can work around them with JSON string
extensions until they land.

### 5.3 Bridge Translation Certificate

The gap at requirement (2) in Section 4.3 requires a "translation certificate"
that proves the trust-llvm2-bridge's translation preserves semantics:

```
TranslationCertificate:
  source_hash: SHA-256 of MIR body (from tRust)
  target_hash: SHA-256 of tMIR body (from bridge output)
  proves: translate(MIR) = tMIR (structural equivalence)
  method: "syntactic" (Phase 1, scalar subset -- direct mapping)
          or "semantic" (Phase 4+, requires SMT proof)
```

For Phase 1 (scalar-only), the translation is a direct syntactic mapping
(each MIR instruction maps to exactly one tMIR instruction). The certificate
is structural: enumerate the mapping and verify it covers all instructions.
No SMT required.

For Phase 4+ (references, structs, SSA construction via Cytron), the
translation is non-trivial and requires its own semantic equivalence proof.
This is deferred to Phase 5 of the integration plan.

---

## 6. Proof Strength Lattice

The composed proof's strength is the weakest link in the chain. The strength
ordering (from weakest to strongest):

```
Sampled(100K)  <  Exhaustive(8-bit)  <  CegisSynthesis  <  SmtUnsat
     |                   |                    |                |
  BoundedSound         Sound               Sound            Sound
  (statistical)     (for width)          (iterative)      (universal)
```

If tRust proves a function with `ProofStrength::smt_unsat()` (Sound) and LLVM2
proves lowering with `LoweringProofStrength::Exhaustive { width_bits: 8 }` (Sound),
the composed proof is Sound. But if LLVM2 can only achieve `Sampled { 100_000 }`,
the composed proof degrades to `BoundedSound`.

**Implication:** LLVM2's z4 integration is not just a nice-to-have. Without it,
the end-to-end chain degrades to statistical confidence, which undermines the
entire "verified compilation" value proposition. The mock evaluation is acceptable
for development and regression testing, but production certificates require z4.

---

## 7. Concrete Implementation Plan

### Step 1: Define LoweringCertificate (LLVM2, ~200 LOC)

Add `crates/llvm2-verify/src/lowering_certificate.rs`:
- `LoweringCertificate`, `RuleProof`, `LoweringVerification`, `LoweringProofStrength`
- Serialization with serde (JSON)
- Hash computation for tMIR and machine code

### Step 2: Generate Certificates in Compiler (LLVM2, ~150 LOC)

Modify `crates/llvm2-codegen/src/compiler.rs`:
- Replace `ProofCertificate { rule_name, verified }` placeholder with
  `Vec<LoweringCertificate>`
- Wire `LoweringCertificateGenerator` into `Compiler::compile()`
- Add `CertificateConfig` to `CompilerConfig`

### Step 3: Conversion to trust-proof-cert (LLVM2, ~200 LOC)

Add `crates/llvm2-verify/src/trust_cert_compat.rs`:
- `impl LoweringCertificate { fn to_proof_certificate() }`
- `impl From<LoweringProofStrength> for ProofStrength`
- `impl From<LoweringVerification> for VerificationResult`

This requires a dev-dependency on trust-types (for type definitions only).
For Phase 1, the conversion can work against the types defined in LLVM2's
stubs. For Phase 2+, switch to a git dependency on trust-types.

### Step 4: CLI Certificate Output (LLVM2, ~50 LOC)

Enhance `llvm2-cli`:
- `--emit-certs` flag to write `<output>.proof.json`
- JSON format: array of LoweringCertificate
- Parseable by tRust's bridge for composition

### Step 5: Propose Upstream Changes (tRust, ~20 LOC)

File issue on trust-proof-cert:
- Add `ChainStepType::MachineLowering`
- Add optional `binary_hash: Option<[u8; 32]>` to ProofCertificate
- Add `StatisticalSampling` to ReasoningKind

### Step 6: Integration Test (Cross-Repo, ~100 LOC)

End-to-end test:
1. tRust verifies `fn add(a: i32, b: i32) -> i32 { a + b }` -> ProofCertificate
2. Bridge translates to tMIR -> LoweringCertificate
3. LLVM2 compiles + verifies -> LoweringCertificate
4. Convert LoweringCertificate to ProofCertificate
5. Compose source + lowering certificates
6. Verify composed proof is Sound

---

## 8. Open Questions

1. **Granularity of lowering certificates.** One certificate per function
   (containing all rule proofs) or one certificate per lowering rule?
   Per-function is recommended for composability with tRust's per-function
   source certificates. Per-rule proofs live inside as `RuleProof` entries.

2. **Optimization certificates.** LLVM2 runs optimization passes (DCE, peephole,
   constant folding) between lowering and encoding. Each pass should ideally
   have its own certificate proving the transformation preserves semantics.
   For Phase 1, treat optimization as part of lowering (one combined certificate).
   For Phase 5, split into per-pass certificates in the chain.

3. **Register allocation certificates.** RA introduces spills and reloads that
   are not present in the pre-RA IR. Proving RA correctness requires showing
   that spilled values are correctly restored. This is a separate verification
   problem from instruction lowering and needs its own certificate type.

4. **Incremental re-verification.** When a function changes, which certificates
   are invalidated? Hash-based: if tMIR hash changes, lowering certificate is
   invalid. If source changes but tMIR is unchanged (e.g., comment-only change),
   lowering certificate remains valid. trust-proof-cert's `ChangeKind::BodyOnly`
   vs `ChangeKind::SpecChanged` distinction applies here.

5. **Certificate size.** z4 UNSAT proofs can be large (MBs for complex
   formulas). Should certificates include raw proof traces or just hashes?
   Recommended: hashes by default, traces opt-in via `CertificateConfig`.

---

## 9. References

- CompCert: Leroy, "Formal verification of a realistic compiler" (CACM 2009)
- Alive2: Lopes et al., "Alive2: Bounded Translation Validation for LLVM" (PLDI 2021)
- tRust proof certs: `~/tRust/crates/trust-proof-cert/` (23K LOC)
- tRust certificate chain: `~/tRust/crates/trust-proof-cert/src/chain.rs`
- tRust composition: `~/tRust/crates/trust-proof-cert/src/composition/`
- tRust verification result: `~/tRust/crates/trust-types/src/result.rs`
- LLVM2 ProofObligation: `crates/llvm2-verify/src/lowering_proof.rs`
- LLVM2 compiler API: `crates/llvm2-codegen/src/compiler.rs`
- LLVM2 bridge design: `designs/2026-04-16-trust-llvm2-bridge.md`
- LLVM2 transval alignment: `designs/2026-04-16-transval-alignment.md`
