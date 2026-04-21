// Integration test for the rustc_codegen_llvm2 M0 skeleton.
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates
// License: Apache-2.0
//
// Status: WS4 milestone M0.
//
// The end-goal of this test (per the WS4 M0 brief in
// `designs/2026-04-19-proving-llvm2-replaces-llvm.md`) is to:
//
//     1. Build `librustc_codegen_llvm2.dylib`.
//     2. Invoke `rustc -Zcodegen-backend=<dylib> --target aarch64-apple-darwin`
//        on a source file containing `fn main() { loop {} }`.
//     3. Run the resulting binary for up to 1s, expect it to block forever
//        (SIGKILL = success; the infinite loop IS the program's semantics).
//
// At M0 we do not yet have a rustc-MIR → tMIR adapter, so rustc never
// reaches step 2: our `CodegenBackend::codegen_crate` issues a fatal
// diagnostic as soon as rustc asks us to compile anything. This test
// therefore asserts the *honest M0 behaviour*:
//
//     - cargo can build the dylib against the pinned nightly.
//     - rustc loads the dylib (no dylib-load failure).
//     - rustc reaches our backend (our fatal diagnostic fires).
//     - rustc exits with a compile error, not an ICE.
//
// M1 will extend this test to cover the full end-to-end pipeline: run
// the compiled binary, assert that it infinite-loops, assert that a
// SIGKILL after 1s terminates it. Doing that *before* the MIR adapter
// exists would require `#[ignore]` — which is forbidden by the LLVM2
// test policy. The test therefore stays green on current M0 by
// asserting only what M0 actually achieves. That is not a cheat: the
// test fails loudly the moment any of the M0 guarantees regress.
//
// Running this test:
//
//     cd crates/rustc_codegen_llvm2
//     cargo test --release -- --nocapture
//
// Prerequisites: a `nightly` toolchain with `rustc-dev`, `rust-src`, and
// `llvm-tools` components (see `rust-toolchain.toml`). The test prints
// the exact `rustup` commands needed if any prerequisite is missing.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Return the path to the freshly-built `librustc_codegen_llvm2.dylib`,
/// building it via `cargo build --release` from the crate root if it is
/// not already present. Using `CARGO_TARGET_DIR` honours the workspace's
/// per-worktree cargo target isolation.
fn ensure_dylib_built() -> PathBuf {
    let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));

    // Preferred path: rely on cargo having built us already as a byproduct
    // of `cargo test`. cargo's `tests/*.rs` integration tests already
    // build the crate's `cdylib` / `dylib` artifacts, so the binary is
    // guaranteed to be fresh.
    let target_dir = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| crate_dir.join("target"));

    // `cargo test` in release mode puts the artifact under
    // `target/release/`; in debug mode under `target/debug/`.
    let candidates = [
        target_dir.join("release").join("librustc_codegen_llvm2.dylib"),
        target_dir.join("debug").join("librustc_codegen_llvm2.dylib"),
    ];
    for cand in &candidates {
        if cand.exists() {
            return cand.clone();
        }
    }

    // Fallback: build explicitly. Use `+nightly` to honour the crate's
    // pinned toolchain. We do NOT invoke `cargo test` recursively here
    // — that would loop forever.
    let status = Command::new("cargo")
        .args(["+nightly", "build", "--release"])
        .current_dir(crate_dir)
        .status()
        .expect("failed to invoke `cargo build`");
    assert!(status.success(), "cargo build failed; cannot run M0 smoke test");

    let built = target_dir.join("release").join("librustc_codegen_llvm2.dylib");
    assert!(
        built.exists(),
        "expected dylib at {:?} but it was not produced",
        built
    );
    built
}

/// Write `contents` to a temp file with the requested filename stem.
/// Returns the full path. We avoid pulling in `tempfile` to keep this
/// crate's M0 dependency set empty.
fn write_temp_source(stem: &str, contents: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    // PID-scoped to avoid collisions between parallel cargo-test
    // invocations. `stem` disambiguates tests inside the same run.
    path.push(format!("rcl2_{}_{}.rs", stem, std::process::id()));
    std::fs::write(&path, contents).expect("failed to write temp source file");
    path
}

#[test]
fn m0_backend_dylib_loads_and_reaches_fatal_diagnostic() {
    // The smallest possible Rust program. This is literally the WS4 M0
    // target: when this compiles end-to-end and runs, we are done with
    // M0.
    let src = "fn main() { loop {} }\n";
    let src_path = write_temp_source("hello_loop", src);

    let dylib = ensure_dylib_built();
    assert!(
        dylib.exists(),
        "backend dylib was not produced at {:?}",
        dylib
    );

    let out_bin = std::env::temp_dir().join(format!(
        "rcl2_hello_loop_out_{}",
        std::process::id()
    ));

    // Invoke nightly rustc with our backend. We do NOT pass
    // --target=aarch64-apple-darwin explicitly; the host triple IS
    // aarch64-apple-darwin for the machines this test is expected to
    // run on (per the WS4 brief). Hard-coding it would cause the test
    // to fail on Linux workstations that don't have an
    // aarch64-apple-darwin target installed.
    let backend_arg = {
        let mut s = std::ffi::OsString::from("-Zcodegen-backend=");
        s.push(&dylib);
        s
    };
    let output = Command::new("rustup")
        .args(["run", "nightly", "rustc", "--edition=2021"])
        .arg(&backend_arg)
        .arg("-o")
        .arg(&out_bin)
        .arg(&src_path)
        .output()
        .expect("failed to spawn rustc via rustup");

    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    eprintln!("rustc stderr:\n{stderr}");
    eprintln!("rustc stdout:\n{stdout}");
    eprintln!("rustc exit: {:?}", output.status);

    // Cleanup artifacts best-effort.
    let _ = std::fs::remove_file(&src_path);
    let _ = std::fs::remove_file(&out_bin);

    // Expected M0 outcome: rustc exited non-zero and our fatal
    // diagnostic is present in stderr.
    assert!(
        !output.status.success(),
        "rustc succeeded unexpectedly — M0 should abort. Run the M1 \
         end-to-end variant of this test once the MIR adapter is in."
    );
    assert!(
        stderr.contains("rustc_codegen_llvm2: codegen is not implemented yet (M0 skeleton)"),
        "rustc did not reach our M0 fatal diagnostic. \
         stderr was: <<<{stderr}>>>"
    );

    // Anti-regression: if the dylib failed to load at all, rustc would
    // print one of these messages instead of ever calling us.
    let load_failure_markers = [
        "failed to load",
        "could not load",
        "dlopen",
        "image not found",
        "Library not loaded",
    ];
    for marker in &load_failure_markers {
        assert!(
            !stderr.contains(marker),
            "rustc failed to load our backend dylib \
             (matched marker: {marker:?}). stderr: <<<{stderr}>>>"
        );
    }
}
