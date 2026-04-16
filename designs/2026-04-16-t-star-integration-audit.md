# t* Stack Integration Audit

**Date:** 2026-04-16
**Author:** Andrew Yates <ayates@dropbox.com>
**Status:** Complete

## Purpose

This document summarizes the findings from a complete audit of the t* stack repositories and their implications for LLVM2. The audit examined every repo in the t* ecosystem to determine: (1) what actually exists, (2) how the pieces connect, and (3) what LLVM2 needs to do next.

## t* Stack Overview

The t* stack is larger and more mature than its minimal tMIR repo suggests. Eight repositories form the ecosystem:

| Repo | LOC | Stage | Description |
|------|-----|-------|-------------|
| tRust | 508K | M1 done, M2 ~90% | Full rustc fork with native verification pass. 34 crates, 13,707 tests. Uses `trust-types::VerifiableFunction` as its IR |
| tMIR | 844 | Minimal | Standalone IR definition. Less complete than LLVM2's stubs (1,714 LOC, 19 tests) |
| sunder | 525K | Building | Deductive verification (Creusot clone) using z4. Source-level contracts |
| certus | 532K | v1.0 done | Verus clone with proof certificates. 5,800+ tests |
| zani | 253K | Active | Kani fork using z4 for bounded model checking |
| z4 | large | Active | SMT solver. 46 crates, 11 theory solvers. QF_BV 100%, QF_LIA 94% |
| tSwift | 138K | Preview | Swift SIL verification frontend |
| tC | 1K | Preview | C verification tool (extremely early) |

**Total ecosystem:** ~1.96M LOC of Rust, all using z4 as the shared verification backend.

## Key Architectural Findings

### Finding 1: tRust Does NOT Produce tMIR

This is the most important finding. tRust uses its own `trust-types::VerifiableFunction` representation (a place-based MIR) rather than the tMIR format that LLVM2 expects. The tMIR bridge has been designed (a 924-line LaTeX document exists at `tRust/designs/2026-04-14-trust-llvm2-integration.tex`) but has not been implemented.

**Implication:** There is no producer of tMIR today. LLVM2 can compile tMIR to machine code, but nothing generates tMIR for it to consume. The trust-llvm2-bridge is the critical path to connecting tRust and LLVM2.

### Finding 2: tMIR Is Bypassed for Rust Verification (Pipeline v2)

In the current tRust architecture (Pipeline v2), the verification pass calls `zani-lib` and `sunder-lib` directly from within the compiler. tMIR is not on this path. tMIR is retained only for the multi-language path (tSwift, tC), where a common IR is needed to share LLVM2 as a backend.

**Implication for LLVM2:** The immediate integration path is tRust -> bridge -> LLVM2, without tMIR as an intermediary. The tMIR-based path becomes relevant only when tSwift and tC mature.

### Finding 3: LLVM2's Stubs Are More Complete Than Real tMIR

A side-by-side comparison reveals that LLVM2's stub tMIR implementation (1,714 LOC, 19 tests) exceeds the real tMIR repo (844 LOC, 0 tests) in functionality:

**LLVM2 stubs have, real tMIR lacks:**
- Atomic operations (5 variants: Load, Store, CmpXchg, Fence, RMW)
- Proof annotations (11 variants: Pure, ValidBorrow, InBounds, etc.)
- InstrNode wrapper with metadata and proof annotation attachment
- Signed/unsigned opcode distinction (SDiv vs UDiv, SRem vs URem, etc.)
- Builder API for programmatic construction
- JSON reader for wire format deserialization

**Real tMIR has, LLVM2 stubs lack:**
- Tuple, Enum, Never types
- Global definitions

**Implication:** The stubs should be upstreamed into the real tMIR repo, and the real tMIR repo's additions (Tuple, Enum, Never, Globals) should be incorporated into the stubs. The two should converge to a single source of truth. Filed as existing issues #257, #258, #261 for individual gaps; a meta-issue for upstreaming is needed.

### Finding 4: No Code to Port from sunder/certus/zani into LLVM2

Despite their size (525K, 532K, 253K LOC respectively), these repos solve a fundamentally different problem than LLVM2:

- **sunder:** Source-level deductive verification (Hoare logic, separation logic). Proves function contracts hold.
- **certus:** Source-level ownership-aware verification. Proves user specifications (#[requires], #[ensures]).
- **zani:** Bounded model checking at MIR level. Explores concrete execution paths.
- **LLVM2:** Machine-level lowering verification. Proves tMIR semantics are preserved through compilation.

The only shared dependency is z4. The bitvector encoding patterns in sunder-z4 and certus-core/encoder are worth studying for implementation ideas, but no code can be directly ported.

### Finding 5: z4 Is Now Wired into LLVM2

The z4 integration work is substantially complete:
- Bridge code fixed (28+ API mismatches between z4's actual API and the original z3-based bridge)
- 158 z4-specific tests pass
- Native API bridge operational (not just subprocess-based SMT-LIB2)

**Remaining gaps:**
- Quantifier proofs need non-QF logic (z4's QF_ABV is insufficient for memory model proofs with forall/exists)
- UF `apply()` needs `FuncDecl` refactor in z4_bridge
- FP encoding issues (fp_const_from_bits, bv_to_fp conversion)

**Upstream z4 wish list filed:**
- z4#8590: `fp_const_from_bits` API
- z4#8591: `Term::sort()` accessor
- z4#8592: QF_ABVFP combined logic
- z4#8593: `bv_const` from u64

### Finding 6: LLVM2 End-to-End Pipeline Works

The full AArch64 pipeline is operational:
- ISel (instruction selection from tMIR)
- Optimization (DCE, constant folding, peephole, CSE, LICM, copy propagation)
- Register allocation (linear scan + greedy)
- Frame lowering and ABI compliance
- Binary encoding (AArch64 instruction encoding)
- Mach-O object file emission
- Linking and execution on Apple Silicon

Verified functions: add, sub, max, factorial — all compile, link, and execute correctly.

## Integration Path

### Interim (Now)

```
tRust (VerifiableFunction)
  |
  v
trust-llvm2-bridge (in tRust repo)
  |  translates VerifiableFunction -> tMIR JSON wire format
  v
LLVM2 CLI (consumes tMIR JSON)
  |
  v
Verified AArch64 binary
```

The bridge crate lives in the tRust repo because it depends on tRust's `trust-types`. It emits JSON in the format LLVM2's stub `tmir-func` JSON reader expects. This is the critical path item.

### Long-term (When tSwift/tC Mature)

```
tRust  --> tMIR (real repo, upstreamed from stubs) --> LLVM2
tSwift --> tMIR                                     --> LLVM2
tC     --> tMIR                                     --> LLVM2
```

The real tMIR repo becomes the single source of truth. All frontends emit tMIR. LLVM2 switches from stubs to the real tMIR crate dependency.

## LLVM2 Current Health

As of this audit:

| Metric | Value |
|--------|-------|
| Total LOC | 126K Rust |
| Crates | 6 + CLI |
| Tests passing | 3,389+ (3 failures fixed during audit) |
| Verification proofs | 960+ (now backed by real z4 SMT solving) |
| Targets | AArch64 (functional), x86-64 (scaffolding) |
| E2E status | Compile, link, execute on Apple Silicon |

## Proof Certificate Chain (Design Gap)

There is currently no mechanism to compose proofs across the stack:

1. **tRust** produces `TrustProofResults` with per-obligation `TrustDisposition` (Trusted/Certified/Failed/Unknown) via its `trust-proof-cert` crate (23,211 LOC).
2. **LLVM2** produces its own lowering proofs via z4.
3. **No connection** between these two proof layers.

For the vision of "verification chain from source to binary is unbroken," a certificate chain design is needed that composes:
- Source-level proofs (tRust/certus/sunder) certifying that the program meets its specification
- Lowering proofs (LLVM2) certifying that the machine code faithfully implements the tMIR

This is a design problem, not an implementation problem yet. The certificate infrastructure exists on both sides; the composition protocol does not.

## Cross-Reference: Machine Semantics

Two independent implementations of AArch64 instruction semantics exist:
- `tRust/crates/trust-machine-sem/` (5,655 LOC) — used for translation validation in tRust
- `LLVM2/crates/llvm2-verify/src/aarch64_semantics.rs` — used for lowering proof verification

These should be cross-referenced to ensure consistency. Long-term, they should be unified or one should depend on the other to prevent divergence.

## Action Items

| Priority | Action | Issue |
|----------|--------|-------|
| P1 | Implement trust-llvm2-bridge in tRust repo | New (filed) |
| P2 | Upstream LLVM2 stub improvements into real tMIR | New (filed) |
| P2 | Design proof certificate chain | New (filed) |
| P2 | Add CLI E2E test with JSON fixture | #256 (existing) |
| P2 | Define LLVM2 library API for tRust consumption | #259 (existing) |
| P3 | Cross-reference trust-machine-sem and llvm2-verify semantics | New (filed) |
| P3 | Resolve tMIR operand model differences | #257 (existing) |
| P3 | Add missing signed/unsigned distinction to real tMIR | #258 (existing) |
| P3 | Resolve tMIR type system representation differences | #261 (existing) |

## References

- tRust codegen integration point: `~/tRust/crates/trust-driver/src/codegen.rs`
- tRust bridge design: `~/tRust/designs/2026-04-14-trust-llvm2-integration.tex`
- tRust proof certificates: `~/tRust/crates/trust-proof-cert/`
- tRust machine semantics: `~/tRust/crates/trust-machine-sem/`
- LLVM2 tMIR adapter: `/Users/ayates/LLVM2/crates/llvm2-lower/src/tmir_adapter.rs`
- LLVM2 stubs: `/Users/ayates/LLVM2/stubs/`
- LLVM2 verification: `/Users/ayates/LLVM2/crates/llvm2-verify/`
- LLVM2 z4 bridge: `/Users/ayates/LLVM2/crates/llvm2-verify/src/z4_bridge.rs`
- Real tMIR repo: `~/tMIR/`
- z4 solver: `~/z4/`
- Sister repo survey: Issue #262
- tRust backend contract: Issue #259
