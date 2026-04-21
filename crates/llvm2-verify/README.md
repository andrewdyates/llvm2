# llvm2-verify

<!-- Andrew Yates <andrewyates.name@gmail.com> -->
<!-- Copyright 2026 Andrew Yates | License: Apache-2.0 -->

Verification testing for the LLVM2 compiler backend. Every tMIR-to-machine
lowering rule, every optimization pass, and every instruction encoder has a
proof obligation that is discharged on `cargo test`.

## Two verification lanes

LLVM2 verifies lowering rules in two complementary lanes. Both run on
`cargo test -p llvm2-verify`; only the second requires a feature flag.

### 1. Default lane — exhaustive evaluation + statistical sampling

`cargo test -p llvm2-verify` (no feature flag) runs every proof through
`verify_by_evaluation_with_config` (`src/lowering_proof.rs`). This path
encodes the tMIR and machine semantics as `SmtExpr` trees and then
*evaluates* both sides on concrete inputs, checking bit-exact equality.

| Width | Strategy | Confidence |
|-------|----------|------------|
| ≤ 8 bits | Exhaustive (all 2^n inputs) | **Formal** — every input space point checked |
| 32 / 64 bits | 100 000 pseudorandom samples | **High-confidence**, not formal |

The default lane is fast (no SMT solver link), always on, and catches the
vast majority of regressions. It is the **authoritative pass/fail signal**
for LLVM2 today.

Relevant entry points:

- `tests/full_proof_suite.rs::full_proof_suite_all_pass` — runs every
  registered proof in `ProofDatabase`.
- `src/verify.rs::VerificationRunner::run_all()` — library-level driver.
- `src/lowering_proof.rs::check_single_point` — the per-sample checker.
  Uses `EvalResult::semantically_equal` (not `==`) so IEEE-754 NaN
  inputs do not produce spurious counterexamples (#388).

### 2. SMT-proof lane — feature-gated z4

`cargo test -p llvm2-verify --features z4-prove` enables the
`tests/z4_prove_smoke.rs` lane, which dispatches 8- and 16-bit proof
obligations to the **z4** SMT solver via `verify_with_z4_api`
(`src/z4_bridge.rs`). z4 reasons symbolically in the theory of
bitvectors, so it proves the obligation over *every* input without
enumerating the input space.

Today the z4-prove smoke lane covers 17 obligations: i8/i16 add/sub/and/
or/xor, i8 neg/shl/lshr/ashr, and i8 bic/orn (see #423, #424, #425,
#426). The full 300+ obligation suite remains on the eval-based lane
until the z4 default-on flip (#407 Task 7), which is blocked on the
Phase-2 overhead profile (#329 Phase 2).

Key invariants of the z4 lane:

- **Default timeout: 30 s per obligation** (`DEFAULT_Z4_TIMEOUT_MS`).
  Override with `LLVM2_Z4_TIMEOUT_MS=<ms>` — `0` disables.
- **Timeout is a proof failure, not a silent pass** (#389). The
  `assert_verified` helper in `z4_prove_smoke.rs` rejects
  `Z4Result::Timeout`, `CounterExample`, and `Error` — only `Verified`
  counts. No silent "good enough".
- **StableHasher-keyed result cache** (`src/z4_cache.rs`) makes the
  second run effectively free on unchanged obligations. Keyed on the
  SMT-LIB2 bytes, the `Z4Config`, and the solver build hash.

## Running

```sh
# Default lane (fast — no SMT linkage)
cargo test -p llvm2-verify

# SMT lane (opts in to z4; links the z4 crate)
cargo test -p llvm2-verify --features z4-prove

# Just the z4 smoke suite
cargo test -p llvm2-verify --features z4-prove --test z4_prove_smoke

# Override the z4 per-obligation timeout
LLVM2_Z4_TIMEOUT_MS=120000 cargo test -p llvm2-verify --features z4-prove
```

Most CI runs the default lane. CI lanes that need formal SMT confidence
should enable `--features z4-prove`. The two lanes prove complementary
properties — `Verified` from z4 is a stronger claim than exhaustive
evaluation at 32/64-bit, but eval-based verification catches regressions
the instant they land (no solver cost, no feature flag).

## Emitting proof artifacts

`llvm2-cli --emit-proofs=<dir>` writes one SMT-LIB2 file and one z4
certificate per discharged obligation. Downstream consumers (tla2, tRust)
import these to chain LLVM2's proofs into their own verification.
See crates/llvm2-cli/src/emit_proofs.rs.

## Known limitations

- **Full SMT proof of 300+ obligations is not default-on.** Gated by
  the #329 Phase-2 overhead profile; #407 Task 7 is the flip.
- **Statistical sampling at 32/64 bits is not formal.** 100 000 random
  inputs is high-confidence but not a proof. z4-prove closes this at
  the cost of solver-link time.
- **Branch-relaxation lowering has no proof obligation yet** (#407
  Task 4 partial).

## References

- designs/2026-04-13-verification-architecture.md — top-level design.
- designs/2026-04-14-z4-integration-guide.md — z4 bridge architecture.
- reports/2026-04-19-z4-coverage-survey.md — which `ProofCategory` runs
  on which lane.
- reports/2026-04-19-407-z4-default-on-audit.md — #407 Phase-0 status.
- evals/results/proofs/2026-04-19.json — per-obligation z4 discharge
  record for the smoke suite.
