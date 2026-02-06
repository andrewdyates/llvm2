<!-- Verified: 488b933 | 2026-02-01T00:00:00Z | [R]3 -->

# Self-Verification Pattern

> Verification-capable tools should verify themselves.

## Overview

Multiple dropbox-ai-prototypes repos have adopted **self-verification** patterns - using their
own verification capabilities on their own codebases. This document establishes
self-verification as an org-wide best practice.

## Why Self-Verification Matters

1. **Dogfooding**: Proves the tool works on real code (its own)
2. **Confidence**: If a verifier can't verify itself, why trust it for others?
3. **Documentation**: Proofs serve as executable specifications
4. **Early Bug Detection**: Self-verification catches internal bugs before users do

## Current Adoption

| Project | Pattern | Coverage | Tracking |
|---------|---------|----------|----------|
| **z4** | Kani proofs for internals | 200 proofs across 22 files | - |
| **lean5** | Self-verification Phase 4 | 7/8 core functions constructive | lean5#371 |
| **gamma-crown** | Soundness provenance | Distinguishes proofs from heuristics | - |
| **zani** | z4 bindings verification | In progress | zani#982, zani#1111 |

Source: dashnews#211, dashnews#216

## Implementation Guidelines

### Kani-Capable Rust Repos

Add proofs for internal invariants using the REQUIRES/ENSURES pattern:

```rust
/// REQUIRES: self.data.len() <= MAX_SIZE
/// ENSURES: Result satisfies invariant XYZ
fn process(&self) -> Result<Output, Error> {
    // ...
}

#[cfg(kani)]
#[kani::proof]
fn verify_process() {
    let input = kani::any();
    kani::assume(input.data.len() <= MAX_SIZE);
    let result = input.process();
    // Assert postconditions
}
```

### Lean Repos

Prove core functions constructively - avoid axioms for fundamental operations:

- Type checker should verify its own typing rules
- Parser should parse its own grammar specification
- Evaluator should evaluate its own semantics

### SMT-Based Tools

Verify solver invariants with the solver itself:

- DPLL invariants proven with solver's own SAT capabilities
- Theory propagation correctness verified symbolically
- Conflict analysis soundness proven

## Maturity Levels

| Level | Description | Example |
|-------|-------------|---------|
| **L0** | No self-verification | Most repos |
| **L1** | Some internal proofs | Basic invariant checks |
| **L2** | Comprehensive coverage | z4 (200 proofs) |
| **L3** | Full self-verification | lean5 Phase 4 goal |

**Target:** L2 for verification repos, L1 for infrastructure.

## When to Adopt

Self-verification is recommended when:

1. **Tool is verification-capable** (provers, type checkers, SMT solvers)
2. **Internal correctness matters** (bugs would undermine trust)
3. **Tool is mature enough** (API stable, time to invest)

Not recommended for:

- Early-stage prototypes (API churn invalidates proofs)
- Tools without verification capabilities
- Repos where testing provides sufficient confidence

## References

- dashnews#211: Cross-org pattern observation
- dashnews#216: lean5 Phase 4 progress
- z4#1158: zani verification harnesses in z4
- lean5#371: Self-verification tracking issue
