# rustc_codegen_llvm2

**Status:** WS4 Milestone M0 — skeleton only. The backend dylib loads
into nightly rustc and fails gracefully with a fatal diagnostic on any
real codegen call.

This crate is the rustc-side of Workstream 4 in
designs/2026-04-19-proving-llvm2-replaces-llvm.md.
Its north-star metric is the **rustc UI test pass-rate** — the single
number we put on every slide to answer "can LLVM2 replace LLVM for
Rust?".

The design template is [`rustc_codegen_cranelift`][cg_clif]. That crate
is the proof that a third-party codegen backend for nightly rustc is a
tractable engineering target, and we deliberately mirror its structure
so we can copy its bugfixes as rustc's internal traits evolve.

[cg_clif]: https://github.com/rust-lang/rustc_codegen_cranelift

## Why this crate is outside the LLVM2 workspace

The LLVM2 workspace targets stable-ish Rust and compiles without
`rustc_private`. This crate:

- Links against rustc's internal crates (`rustc_codegen_ssa`,
  `rustc_middle`, `rustc_session`, …) via `#![feature(rustc_private)]`.
- Pins a nightly toolchain via its own `rust-toolchain.toml`.
- Is built as a `dylib` (not `cdylib`) so rustc can `dlopen` it via
  `-Zcodegen-backend=<path>`.

Mixing those constraints with the rest of the workspace would drag the
whole repo onto a nightly pin. Keeping this crate isolated (via an
empty `[workspace]` table in its `Cargo.toml`) is how cranelift does
it and it's the pragmatic choice.

## Roadmap (M0 → M4)

| Milestone | Scope                                                                           | Rough UI pass-rate |
|-----------|---------------------------------------------------------------------------------|--------------------|
| **M0**    | **Skeleton compiles. Backend dylib loads. Fatal diagnostic on codegen.** (done) | 0%                 |
| M1        | Integer arithmetic, control flow, function calls.                               | ~15%               |
| M2        | Structs, enums (tagged unions), references, borrows.                            | ~50%               |
| M3        | Generics post-monomorphization, trait objects, drop glue.                       | ~85%               |
| M4        | `core` / `alloc` / `std` intrinsics, inline asm, panic machinery.               | remainder          |

See the design doc's WS4 section for the full rationale. Each milestone
lands with an updated UI pass-rate measurement recorded under
`evals/results/rustc-backend/`.

## How to test locally

Prerequisites:

- `rustup` installed.
- A nightly toolchain matching this crate's `rust-toolchain.toml`.
  We pin `channel = "nightly"` (the latest stable-blessed nightly on
  the host). As of 2026-04-20 PM, that resolves to
  `rustc 1.97.0-nightly (e22c616e4 2026-04-19)`. **The crate MUST be
  rebuilt whenever rustup bumps `nightly`, because `rustc_codegen_ssa`
  has an unstable ABI.** If you see `E0050`/`E0046` / `E0432` errors
  from this crate, that's the symptom of the trait drifting under us;
  re-read the current trait shape from
  $(rustup run nightly rustc --print sysroot)/lib/rustlib/rustc-src/rust/compiler/rustc_codegen_ssa/src/traits/backend.rs
  and re-shape `impl CodegenBackend` in `src/lib.rs` to match.
- The `rustc-dev`, `rust-src`, and `llvm-tools` components installed on
  that toolchain. The smoke-test script will print the exact `rustup`
  command if any of them is missing.

### Option A: shell smoke test (script)

From the repo root:

```bash
./scripts/test_rustc_codegen_llvm2.sh
```

This will:

1. Verify the pinned nightly is installed with `rustc-dev`.
2. Build `librustc_codegen_llvm2.dylib` in release mode.
3. Invoke `rustc -Zcodegen-backend=<dylib>` on
   [`examples/hello.rs`](examples/hello.rs) (a `fn main() { loop {} }`).
4. Classify the result into one of:
   - `backend_loaded_but_codegen_unimplemented` — expected M0 outcome.
   - `hello_compiled_and_runs` — expected once M1 lands.
   - `dylib_load_failure` — a regression; M0 is broken.
   - `unexpected_failure` — rustc exited non-zero with an unknown
     error. Record it and investigate.
5. Append the result to evals/results/rustc-backend/<ISO-date>.json.

At M0 the script exits 0 as long as rustc successfully loaded our
dylib and hit our fatal diagnostic. That is a real, meaningful,
measurable result.

### Option B: `cargo test` integration test

From the crate directory:

```bash
cd crates/rustc_codegen_llvm2
cargo test --release
```

This runs `tests/hello_loop.rs`, which builds the dylib, invokes
rustup run nightly rustc --edition=2021 -Zcodegen-backend=<dylib> \
  -o /tmp/rcl2_hello_loop_out_<pid> <fn main() { loop {} }>.rs, and
asserts:

- rustc exits non-zero (M0: no binary is produced).
- stderr contains our fatal diagnostic string (proves rustc reached
  our `codegen_crate`).
- stderr does NOT contain any dlopen / "Library not loaded" /
  "image not found" / "failed to load" markers (proves the dylib
  actually loaded).

When M1 lands, this same test gets extended with a
`Command::spawn` + 1-second wait + `kill` + assert that the binary
produced output nothing and was still running — i.e., the infinite
loop is working. See the top-of-file comment in `tests/hello_loop.rs`
for the exact extension plan.

## rustc UI test harness plan (M1+)

Once M1 lands, we will drive rustc's UI test suite against
`rustc_codegen_llvm2` using the same mechanism cranelift uses:

- Check out the `rust-lang/rust` tree at the commit matching our
  pinned nightly.
- Build a sysroot with `rustc_codegen_llvm2` substituted for
  `rustc_codegen_llvm`.
- Invoke `./x.py test tests/ui` with the substitute sysroot.
- Record pass / fail / ignored counts and diff them against the
  upstream baseline.

Cranelift's scripts/test_rustc.sh and `patches/` directory are the
immediate reference. We will add our equivalents under
`scripts/rustc_ui_harness/` at M1.

## What blocks M1

M1 = "compile `fn main() { let x = 1 + 2; }` through rustc + LLVM2 to
a running binary". The minimum new pieces required:

1. **`tmir-from-rustc-mir` adapter** — a translator from rustc MIR (as
   seen through `rustc_middle::mir`) to tMIR. At M1 this only needs to
   cover: `Local` / `Place`, `Rvalue::Use`, `Rvalue::BinaryOp` for
   integer arith, `StatementKind::Assign`, `TerminatorKind::Return` /
   `Call` / `Goto` / `SwitchInt`, integer-ty layouts. This will live
   in a new crate `crates/rustc_codegen_llvm2_adapter/` (so the
   rustc-private bits are isolated from the LLVM2 workspace) or as a
   module inside this crate.
2. **Driver loop** — a `codegen_crate` implementation that walks the
   monomorphization set (`tcx.collect_and_partition_mono_items`),
   calls the tMIR adapter per function, hands each tMIR module to
   `llvm2-lower`, and collects the emitted object files.
3. **Object file plumbing** — wire `llvm2-codegen`'s Mach-O writer
   output into rustc's expected per-module artifacts
   (`CompiledModule`). Our workspace's Mach-O writer already produces
   valid Mach-O; M1 only needs to route bytes to the right path.
4. **Symbol mangling** — use `rustc_symbol_mangling::symbol_name`
   instead of any ad-hoc scheme, so the linker finds our symbols.
5. **`join_codegen` + `link`** — stub them for a single-CGU crate
   first; cranelift's driver/aot.rs is the reference.
6. **Target sysroot handling** — at M1 we only promise to compile the
   user's crate; rustc will still link against a stock libstd built by
   LLVM. Full sysroot rebuild is M3/M4.

Until those land, this crate intentionally stays a fatal-diagnostic
stub. That is not a limitation — it is the entire point of M0:
everything downstream of "rustc loads our dylib" can now be measured.
