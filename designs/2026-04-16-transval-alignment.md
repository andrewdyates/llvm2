# Translation Validation Alignment: LLVM2 and tRust trust-transval

**Date:** 2026-04-16
**Author:** Andrew Yates <ayates@dropbox.com>
**Status:** Draft
**Part of:** #260 (Align LLVM2 verification with trust-transval)

---

## Overview

This document analyzes how tRust's `trust-transval` crate structures translation
validation, how LLVM2's `llvm2-verify` crate compares, and what changes are
needed to ensure compatibility between the two systems. The goal is an unbroken
verification chain from tRust source through tMIR to machine code.

**References:**
- `~/tRust/crates/trust-transval/` -- tRust translation validation crate
- `crates/llvm2-verify/` -- LLVM2 verification crate
- Pnueli, Siegel, Singerman. "Translation Validation" (TACAS 1998)
- Necula. "Translation Validation for an Optimizing Compiler" (PLDI 2000)
- Lopes et al. "Alive2: Bounded Translation Validation for LLVM" (PLDI 2021)

---

## 1. How trust-transval Works

### 1.1 Architecture

trust-transval validates that compiler transformations (MIR optimizations,
lowering) preserve semantics. It is structured as a layered pipeline:

```
VerifiableFunction (source)  ──┐
                               ├──> SimulationRelationBuilder ──> SimulationRelation
VerifiableFunction (target)  ──┘                                       │
                                                                       ▼
                                                        EquivalenceVcGenerator
                                                                       │
                                                                       ▼
                                                          Vec<RefinementVc>
                                                                       │
                                                                       ▼
                                                        TranslationValidator
                                                        (Router-backed SMT)
                                                                       │
                                                                       ▼
                                                     SmtValidationResult
                                                 (Equivalent | Divergent | Inconclusive)
```

### 1.2 Core Types (from trust-types)

| Type | Purpose |
|------|---------|
| `VerifiableFunction` | Function representation with body, contracts, pre/postconditions |
| `VerifiableBody` | Locals (typed), basic blocks, arg count, return type |
| `BasicBlock` | BlockId + statements + terminator |
| `Statement::Assign` | Place = Rvalue (BinaryOp, UnaryOp, Use, Cast, Ref) |
| `Terminator` | Goto, SwitchInt, Return, Unreachable, Call, Assert, Drop |
| `SimulationRelation` | Maps source blocks -> target blocks, source locals -> target locals |
| `RefinementVc` | Verification condition with TranslationCheck + source/target function names |
| `TranslationCheck` | source_point, target_point, CheckKind, Formula, description |
| `CheckKind` | ControlFlow, DataFlow, ReturnValue, Termination |
| `Formula` | SMT formula AST (Bool, Int, Var, Add, Sub, Mul, Eq, Not, And, Or, etc.) |

### 1.3 Core Theorem

trust-transval proves **refinement**:

```
forall inputs, not UB_MIR(inputs) => Behavior(target, inputs) in Behavior(source, inputs)
```

This is checked by generating VCs of the form `NOT(source_expr == target_expr)`.
If all VCs are UNSAT, the transformation is valid.

### 1.4 Four Categories of VCs

1. **Control Flow Preservation**: For each conditional branch in the source,
   the target's corresponding block must reach the corresponding successor.

2. **Data Flow Preservation**: For each assignment in the source, the target's
   corresponding block computes an equivalent value for the mapped variable.

3. **Return Value Preservation**: If the source returns, the target must also
   return, and `_0` (the return local) must hold an equivalent value.

4. **Termination Preservation**: If the source has loops (back-edges), the
   target must preserve them or validly eliminate them.

### 1.5 Simulation Relation

The `SimulationRelation` maps source program points to target program points:
- `block_map: HashMap<BlockId, BlockId>` -- source block -> target block
- `variable_map: HashMap<usize, usize>` -- source local -> target local
- `expression_map: HashMap<usize, Formula>` -- source local -> target expression

The builder infers this relation automatically (positional for same block count,
terminator-based matching for different counts) or accepts manual overrides.

### 1.6 Optimization Pass Awareness

`OptimizationPassValidator` provides pass-specific strategies:
- **Constant folding**: Builds expression maps for folded constants
- **DCE**: Skips termination checks (DCE only removes code)
- **Copy propagation**: Standard refinement check
- **Generic**: Falls back to general refinement

### 1.7 Solver Backend

`TranslationValidator` uses `trust-router` for SMT backend dispatch. The router
supports MockBackend (always available) and z4 backend (when configured). The
validator also performs straight-line concrete evaluation as a fast path before
dispatching to the solver.

---

## 2. How LLVM2 Verification Compares

### 2.1 Architecture

LLVM2's verification focuses on **per-rule equivalence**: proving that individual
lowering rules (tMIR instruction -> AArch64 instruction sequence) are correct.

```
SmtExpr (tMIR semantics)    ──┐
                               ├──> ProofObligation ──> verify_by_evaluation()
SmtExpr (AArch64 semantics) ──┘                              │
                                                              ▼
                                                    VerificationResult
                                                   (Valid | Invalid | Unknown)
```

### 2.2 Core Types

| Type | Purpose |
|------|---------|
| `ProofObligation` | Pairs tMIR and AArch64 SmtExprs with inputs and preconditions |
| `SmtExpr` | Bitvector expression AST (self-contained, no external SMT lib dependency) |
| `VerificationResult` | Valid, Invalid{counterexample}, Unknown{reason} |
| `VerificationStrength` | Exhaustive, Statistical{sample_count}, Formal |
| `VerificationReport` | Aggregated results with per-category breakdowns |
| `Verifier` | Unified entry point running all proof categories |

### 2.3 Verification Scope

LLVM2 currently verifies:
- Arithmetic lowering (add, sub, mul, neg at 32/64-bit)
- NZCV flag correctness (4 flags, 10 comparison conditions, branches)
- Peephole optimization identities (11+ rules)
- Memory model (load/store equivalence, roundtrip, non-interference, endianness)
- Optimization passes (constant folding, copy prop, CSE/LICM, DCE, CFG)
- NEON SIMD semantics and lowering
- GPU (Metal) and ANE semantics

---

## 3. Key Differences

| Aspect | trust-transval | llvm2-verify |
|--------|---------------|--------------|
| **Granularity** | Whole-function refinement | Per-rule equivalence |
| **IR level** | MIR -> optimized MIR | tMIR -> AArch64 machine instructions |
| **Relation** | SimulationRelation (block/variable maps) | Direct expression pairing |
| **VC categories** | ControlFlow, DataFlow, ReturnValue, Termination | Single equivalence check per rule |
| **Formula repr** | `Formula` (trust-types: Int, Bool, Var, BinOps) | `SmtExpr` (bitvector-native: BvAdd, BvSub, Extract, etc.) |
| **Solver** | trust-router (MockBackend or z4) | Mock evaluation (exhaustive/statistical) |
| **CFG analysis** | Yes (block successors, back-edge detection) | No (per-rule, not per-function) |
| **Optimization awareness** | Yes (pass-specific strategies) | Partial (separate proof modules per pass) |

### 3.1 Complementary Not Conflicting

The two systems verify different compilation stages:

```
tRust source
    │
    ▼
  MIR (pre-optimization)
    │  ← trust-transval verifies THIS transformation
    ▼
  MIR (post-optimization)
    │
    ▼
  tMIR
    │  ← llvm2-verify verifies THIS transformation
    ▼
  AArch64 machine code
```

trust-transval proves MIR optimizations correct. llvm2-verify proves
tMIR-to-machine-code lowering correct. Together they form an end-to-end chain.

### 3.2 The Gap: tMIR Interface

The interface between the two systems is tMIR. Currently:
- trust-transval validates transformations on `VerifiableFunction` (MIR-level)
- llvm2-verify validates lowering from tMIR stubs (not real tMIR types)
- There is no formal connection between trust-transval's output and llvm2-verify's input

---

## 4. Changes Needed for Alignment

### 4.1 Shared Vocabulary (Priority: High)

**Problem:** trust-transval uses `Formula` (from trust-types) for VC formulas.
llvm2-verify uses `SmtExpr` (self-contained bitvector AST). These are
semantically similar but structurally different.

**Solution:** Implement bidirectional conversion between `Formula` and `SmtExpr`.

```rust
// In llvm2-verify: convert trust-types Formula to our SmtExpr
impl From<&Formula> for SmtExpr { ... }

// In llvm2-verify: convert our SmtExpr to trust-types Formula
impl From<&SmtExpr> for Formula { ... }
```

This allows LLVM2 to consume VCs from trust-transval and produce VCs that
trust-transval can validate.

### 4.2 CheckKind Alignment (Priority: High)

**Problem:** trust-transval classifies VCs into four kinds (ControlFlow,
DataFlow, ReturnValue, Termination). LLVM2 has no VC classification beyond
informal category strings ("arithmetic", "memory", "peephole").

**Solution:** Add a `TransvalCheckKind` enum to llvm2-verify that maps to
trust-transval's `CheckKind` where applicable and extends it for machine-specific
concerns. Named `TransvalCheckKind` (not `ProofCategory`) to avoid conflict with
the existing fine-grained `proof_database::ProofCategory` (36 variants):

```rust
/// Translation validation check kind, aligned with trust-transval's CheckKind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TransvalCheckKind {
    DataFlow,        // Maps to trust-transval CheckKind::DataFlow
    ControlFlow,     // Maps to CheckKind::ControlFlow
    ReturnValue,     // Maps to CheckKind::ReturnValue
    Termination,     // Maps to CheckKind::Termination
    InstructionLowering,   // LLVM2-specific
    PeepholeOptimization,  // LLVM2-specific
    MemoryModel,           // LLVM2-specific
    RegisterAllocation,    // LLVM2-specific
    Vectorization,         // LLVM2-specific
}
```

### 4.3 Whole-Function Verification (Priority: Medium)

**Problem:** trust-transval validates whole functions (source vs target) with
simulation relations. LLVM2 only validates individual lowering rules.
`Verifier::verify_transformation()` is a stub returning `Unknown`.

**Solution:** Implement whole-function verification by composing per-rule proofs:

1. Accept a tMIR function and its lowered MachFunction
2. Build a simulation relation mapping tMIR basic blocks to machine blocks
3. For each mapped block pair, compose per-instruction proofs into a block-level
   data-flow VC
4. Generate ControlFlow and ReturnValue VCs analogous to trust-transval

This is the key missing piece for end-to-end validation.

### 4.4 RefinementVc Adapter (Priority: Medium)

**Problem:** trust-transval produces `RefinementVc` (from trust-types). LLVM2
has no way to consume or produce these.

**Solution:** Create an adapter layer that converts between the two systems:

```rust
/// Convert a trust-transval RefinementVc to an LLVM2 ProofObligation.
///
/// This enables LLVM2 to verify VCs produced by trust-transval when
/// they involve tMIR-level semantics that map to machine operations.
pub fn refinement_vc_to_proof_obligation(vc: &RefinementVc) -> ProofObligation {
    ProofObligation {
        name: vc.check.description.clone(),
        tmir_expr: formula_to_smt_expr(&extract_source_expr(&vc.check.formula)),
        aarch64_expr: formula_to_smt_expr(&extract_target_expr(&vc.check.formula)),
        inputs: extract_variables(&vc.check.formula),
        preconditions: vec![],
        fp_inputs: vec![],
    }
}

/// Convert an LLVM2 ProofObligation to a trust-transval-compatible
/// RefinementVc for cross-system validation.
pub fn proof_obligation_to_refinement_vc(
    obligation: &ProofObligation,
    source_fn: &str,
    target_fn: &str,
) -> RefinementVc { ... }
```

### 4.5 Optimization Pass Parity (Priority: Low)

**Problem:** trust-transval has pass-specific strategies for constant folding,
DCE, copy propagation. LLVM2 has separate proof modules for each optimization
but no unified OptimizationPass abstraction.

**Solution:** Not urgent. LLVM2's per-module approach works at the machine level.
However, for future integration, consider adding an `OptimizationPassProver` that
mirrors trust-transval's `OptimizationPassValidator` at the machine level.

### 4.6 Property Checking Parity (Priority: Low)

**Problem:** trust-transval has a `PropertyChecker` that validates four
translation properties (SemanticPreservation, TypePreservation,
TerminationPreservation, MemorySafety). LLVM2 has no equivalent.

**Solution:** Not urgent for code generation verification. Type and memory
safety properties are ensured by tMIR's type system and LLVM2's lowering rules.
Consider adding property checks when whole-function verification is implemented.

---

## 5. Proposed API Compatibility Layer

### 5.1 Module: `llvm2-verify/src/transval_compat.rs`

A new module providing the bridge between the two verification systems:

```rust
//! Compatibility layer between LLVM2 verification and tRust trust-transval.
//!
//! Enables LLVM2 to participate in tRust's end-to-end translation validation
//! pipeline by converting between Formula/SmtExpr representations and
//! RefinementVc/ProofObligation structures.

/// Translation validation check kind, aligned with trust-transval CheckKind.
pub use lowering_proof::TransvalCheckKind;

/// Convert trust-types Formula to LLVM2 SmtExpr.
pub fn formula_to_smt_expr(formula: &Formula) -> SmtExpr { ... }

/// Convert LLVM2 SmtExpr to trust-types Formula.
pub fn smt_expr_to_formula(expr: &SmtExpr) -> Formula { ... }

/// Convert a verification report to trust-transval-compatible format.
pub fn report_to_validation_result(report: &VerificationReport) -> SmtValidationResult { ... }

/// Accept trust-transval RefinementVcs and verify them with LLVM2's engine.
pub fn verify_refinement_vcs(vcs: &[RefinementVc]) -> VerificationReport { ... }
```

### 5.2 Integration Points

```
trust-transval                          llvm2-verify
──────────────                          ────────────
VerifiableFunction (MIR)                ProofObligation (tMIR->AArch64)
     │                                       │
     ▼                                       ▼
RefinementVc ──── transval_compat ────> ProofObligation
     │           formula_to_smt_expr         │
     │           smt_expr_to_formula         │
     │                                       ▼
     │                                  VerificationReport
     │                                       │
     ▼                                       ▼
SmtValidationResult ←── transval_compat ── VerificationReport
```

### 5.3 End-to-End Verification Chain

With the compatibility layer, the full verification chain becomes:

```
tRust source ──► MIR
                  │
         trust-transval validates: MIR → optimized MIR
                  │
                  ▼
            optimized MIR ──► tMIR
                  │
         trust-transval validates: MIR → tMIR (via adapter)
                  │
                  ▼
              tMIR
                  │
         llvm2-verify validates: tMIR → AArch64
         (using per-rule + whole-function proofs)
                  │
                  ▼
         AArch64 machine code (verified)
```

---

## 6. Concrete Changes Implemented

### 6.1 TransvalCheckKind Enum

Added `TransvalCheckKind` to `lowering_proof.rs` to formalize the category
taxonomy and align with trust-transval's `CheckKind`. Named `TransvalCheckKind`
(not `ProofCategory`) to avoid conflict with the existing fine-grained
`proof_database::ProofCategory` enum (36 LLVM2-specific variants).

trust-transval-compatible variants:
- `DataFlow` maps to `CheckKind::DataFlow`
- `ControlFlow` maps to `CheckKind::ControlFlow`
- `ReturnValue` maps to `CheckKind::ReturnValue`
- `Termination` maps to `CheckKind::Termination`

LLVM2-specific extensions:
- `InstructionLowering`, `PeepholeOptimization`, `MemoryModel`,
  `RegisterAllocation`, `Vectorization`

### 6.2 Category Field on ProofObligation

Added an optional `category: Option<TransvalCheckKind>` field to
`ProofObligation` so that each proof obligation can carry its trust-transval
classification. Existing proofs default to `None` (backward compatible).
Future work can populate this field to enable cross-system VC flow.

---

## 7. Concrete Next Steps (Updated 2026-04-16)

Based on the proof certificate chain design (`designs/2026-04-16-proof-certificate-chain.md`)
and the t* stack integration audit, the following concrete steps have been identified.
These are ordered by dependency (each step enables the next).

### 7.1 Formula <-> SmtExpr Conversion (High Priority)

**What:** Implement bidirectional conversion between trust-types `Formula` and
LLVM2's `SmtExpr`.

**Why:** This is the prerequisite for all VC interoperability. Without it, LLVM2
cannot consume trust-transval VCs, and trust-transval cannot validate LLVM2
lowering results.

**Key challenge:** trust-types `Formula` operates on unbounded integers and booleans,
while `SmtExpr` is bitvector-native (fixed-width). The conversion must:
- `Formula::Int` -> `SmtExpr::BvConst` with explicit width parameter
- `Formula::Add/Sub/Mul` -> `SmtExpr::BvAdd/BvSub/BvMul`
- `Formula::Var` -> `SmtExpr::Var` with width from a type environment
- Handle `Formula::Bool` / `Formula::Not/And/Or` as 1-bit bitvectors or as
  separate boolean expressions depending on context

**Approach:** Define a `FormulaContext` that carries bit-width information per
variable. trust-transval's formulas operate on MIR locals with known types, so the
width information is available at VC generation time.

**Deliverable:** `llvm2-verify/src/transval_compat.rs` module with:
```rust
pub fn formula_to_smt_expr(formula: &Formula, ctx: &FormulaContext) -> SmtExpr;
pub fn smt_expr_to_formula(expr: &SmtExpr) -> Formula;
```

### 7.2 Populate TransvalCheckKind on Existing Proofs (High Priority)

**What:** Set the `category` field on ProofObligation across all existing proof
modules. Currently all obligations have `category: None`.

**Why:** Without category tags, LLVM2 cannot generate structured `RuleProof`
entries for the certificate chain, and cannot report verification coverage by
trust-transval CheckKind.

**Scope:** The following modules create ProofObligations and need updates:
- `lowering_proof.rs`: arithmetic, comparison, branch -> `InstructionLowering`
- `nzcv_proof.rs`: flag proofs -> `InstructionLowering`
- `peephole_proof.rs`: rewrite rules -> `PeepholeOptimization`
- `memory_proof.rs`: load/store -> `MemoryModel`
- `dce_proof.rs`, `constfold_proof.rs`, `copyprop_proof.rs`: -> `DataFlow`
- `cse_licm_proof.rs`: -> `DataFlow`
- `cfg_proof.rs`: branch folding -> `ControlFlow`
- `neon_proof.rs`, `vectorization_proof.rs`: -> `Vectorization`

### 7.3 Whole-Function Verification via Block Composition (Medium Priority)

**What:** Implement `Verifier::verify_transformation()` by composing per-rule
proofs into whole-function verification.

**Why:** trust-transval validates whole functions, not individual rules. To
compose certificates, LLVM2 must be able to certify entire function lowerings,
not just individual instruction translations.

**Approach:** For each tMIR function and its lowered MachFunction:
1. Build a block mapping: tMIR BlockId -> MachBlock (positional in Phase 1)
2. For each mapped block pair, verify all instruction lowerings using existing
   per-rule proofs
3. Compose into block-level DataFlow VCs
4. Generate ControlFlow VCs by checking branch target correspondence
5. Generate ReturnValue VC by checking return instruction equivalence
6. Termination: guaranteed for Phase 1 (no loops or bounded loops only)

This creates a per-function `Vec<ProofObligation>` with full TransvalCheckKind
coverage, suitable for bundling into a `LoweringCertificate`.

### 7.4 RefinementVc Adapter (Medium Priority)

**What:** Implement conversion between trust-transval `RefinementVc` and LLVM2
`ProofObligation`.

**Why:** Enables the tRust bridge to pass trust-transval VCs to LLVM2 for
verification (useful when trust-transval validates the MIR-to-tMIR translation
and wants LLVM2's z4 engine as a solver backend).

**Depends on:** 7.1 (Formula/SmtExpr conversion).

### 7.5 LoweringCertificate Generation (High Priority)

**What:** Implement `LoweringCertificate` and `LoweringCertificateGenerator`
as designed in `designs/2026-04-16-proof-certificate-chain.md`.

**Why:** This is the LLVM2-side output of the certificate chain. Without it,
there is nothing to compose with tRust source certificates.

**Depends on:** 7.2 (TransvalCheckKind populated), 7.3 (whole-function
verification for per-function certificates).

### 7.6 trust-proof-cert Compatibility (Medium Priority)

**What:** Implement `to_proof_certificate()` conversion from LoweringCertificate
to trust-proof-cert's ProofCertificate.

**Why:** Enables composition with tRust source certificates via trust-proof-cert's
existing composition module.

**Depends on:** 7.5, upstream addition of `ChainStepType::MachineLowering`.

---

## 8. Gap Assessment: Proof Obligations vs trust-transval

### 8.1 Current Coverage

LLVM2 has 960+ proof tests across 27 modules, but these are **per-rule** proofs.
trust-transval generates **per-function** VCs in four categories. The gap:

| trust-transval Category | LLVM2 Coverage | Gap |
|------------------------|----------------|-----|
| **DataFlow** | Per-instruction: 200+ proofs for arithmetic, comparisons, memory | Per-instruction only; no per-function composition |
| **ControlFlow** | CFG simplification proofs (branch folding, constant folding) | No branch-target-correspondence proof for whole functions |
| **ReturnValue** | Not explicitly verified | No return-value-equivalence proof |
| **Termination** | Not addressed | Not needed for Phase 1 (no loops in scalar subset) |

### 8.2 Key Finding

The fundamental gap is **granularity**, not coverage. LLVM2 has extensive
per-rule proofs but no mechanism to compose them into whole-function certificates.
trust-transval has the whole-function framework but operates at a higher IR level
(MIR, not machine code).

The alignment work is to make LLVM2's per-rule proofs composable into
trust-transval-compatible per-function certificates. This is Steps 7.2-7.3 above.

### 8.3 What Is NOT a Gap

- **Solver compatibility**: Both use z4. LLVM2's z4 bridge is operational (28+
  API mismatches fixed, 158 z4-specific tests pass).
- **Proof strength types**: Both use ProofStrength with reasoning + assurance.
  LLVM2's `LoweringProofStrength` maps cleanly to trust-types' `ProofStrength`.
- **TransvalCheckKind alignment**: Already implemented (Section 6.1).
- **Category field on ProofObligation**: Already implemented (Section 6.2).

---

## 9. Summary

| Aspect | Status | Priority |
|--------|--------|----------|
| Design analysis complete | Done | -- |
| TransvalCheckKind enum | Implemented | High |
| ProofObligation.category field | Implemented | High |
| Formula <-> SmtExpr conversion | Next step (7.1) | High |
| Populate category on all proofs | Next step (7.2) | High |
| LoweringCertificate generation | Next step (7.5) | High |
| Whole-function verification | Next step (7.3) | Medium |
| RefinementVc adapter | Next step (7.4) | Medium |
| trust-proof-cert compatibility | Next step (7.6) | Medium |
| Optimization pass parity | Future | Low |
| Property checking parity | Future | Low |

The two systems are complementary: trust-transval validates MIR transformations,
llvm2-verify validates machine code generation. The fundamental gap is
**granularity** (per-rule vs per-function), not coverage or vocabulary.
The key alignment work is composing LLVM2's per-rule proofs into
trust-transval-compatible per-function certificates. The proof certificate
chain design (`designs/2026-04-16-proof-certificate-chain.md`) provides the
concrete format and composition protocol for connecting these systems.
