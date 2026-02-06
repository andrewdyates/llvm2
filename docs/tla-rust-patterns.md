<!-- Verified: 4ca5435 | 2026-01-30T22:43:34Z | [W]5 -->

# TLA+ to Rust Correspondence Patterns

Author: Andrew Yates <ayates@dropbox.com>

This document captures a reusable pattern for Rust repos that map TLA+ specs to
implementation code. It mirrors the Python-focused guidance in
`docs/tla_correspondence.md`, but uses the z4 project as the reference
implementation for Rust.

## Reference Implementation

Use z4's `z4-tla-bridge` crate as the baseline for how to run TLC from Rust and
interpret outcomes. In the z4 repo, see:

- `crates/z4-tla-bridge/src/lib.rs`
  - `TlcRunner` for programmatic TLC execution
  - `TlcOutcome` enum for success vs. failure classification
  - `TlcViolation` for counterexample traces
  - `TlcErrorCode` taxonomy for programmatic handling

If you need a starting point, browse the crate before adding new abstractions
in your repo.

## TLA_TO_RUST_MAPPING.md Template

Each Rust repo with TLA+ specs should maintain a `docs/TLA_TO_RUST_MAPPING.md`
file that maps the model to the implementation. Recommended structure:

1. **Spec Inventory**
   - Table listing spec modules (`tla/*.tla`), TLC configs (`*.cfg`), and the
     Rust crate(s) that implement them.
2. **State Variable Mapping**
   - Table mapping each TLA+ variable to Rust state and types.
3. **Action Mapping**
   - Table mapping each TLA+ action to Rust functions or methods.
4. **Invariant Mapping**
   - Table mapping each TLA+ invariant to Rust checks or property tests.
5. **Property Test Coverage Summary**
   - List of proptest modules and the invariants/actions they exercise.
6. **External Events and Environment**
   - Table mapping environment events (crash, restart, time passage) to TLA+
     actions.
7. **TLC Configuration and Execution**
   - Location of TLC config files, plus the Rust entry point used to run TLC.

The canonical example is `docs/TLA_TO_RUST_MAPPING.md` in the
[z4 repo](https://github.com/dropbox-ai-prototypes/z4).

## Property Test Naming

Use a consistent naming convention so correspondence is grep-friendly:

- Invariant proptests: `tla_invariant_<invariant_name>`
- Action proptests (if present): `tla_action_<action_name>`

Use the exact TLA+ identifier in the test name to keep grep-based traceability
reliable.

Example:

```rust
proptest! {
    #[test]
    fn tla_invariant_lock_mutex(...) {
        // ...
    }
}
```

## When to Update

Update `TLA_TO_RUST_MAPPING.md` whenever:

- A TLA+ spec changes (new variables/actions/invariants).
- TLC configs change (new or edited `.cfg` files).
- The Rust implementation changes behavior relevant to the model.
- Property tests are added or removed.

## Correspondence Verification

To keep correspondence trustworthy:

1. Update the mapping doc alongside spec or code changes.
2. Run TLC through your Rust wrapper (for example, `TlcRunner`) with the same
   config referenced in `TLA_TO_RUST_MAPPING.md`.
3. Run property tests that encode invariants (for example,
   `cargo test -p <crate> tla_invariant_`).
