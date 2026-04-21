// llvm2-codegen/tests/jit_integration_x86_64.rs - x86-64 JIT smoke test
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Smoke test for the x86-64 JIT path: compile a trivial `const42()` and
// `add(i64, i64) -> i64` function through the x86-64 pipeline (`X86Pipeline`),
// mmap the resulting raw code bytes as executable memory, invoke them as
// `extern "C"` function pointers, and assert the return values.
//
// Part of #467 — Add x86-64 JIT smoke test (#[cfg(target_arch = "x86_64")]-gated)
// Part of #445 — x86-64 end-to-end production hardening
//
// # cfg-gating
//
// This entire file is gated on `#[cfg(target_arch = "x86_64")]`. On
// AArch64 hosts (the primary dev platform) the file is skipped at compile
// time — `cargo check --tests` still validates its syntax via the outer
// file-level `#![cfg]` attribute (rustc parses and attribute-filters the
// contents before type-checking). The test body runs only on x86-64 hosts
// (Linux x86_64, macOS Intel).
//
// # Design choice: bypass JitCompiler::compile_raw
//
// `JitCompiler::compile_raw` (in `llvm2-codegen::jit`) internally calls
// `crate::pipeline::encode_function_with_fixups`, which is an AArch64-only
// encoder keyed off `AArch64Opcode`. Feeding an x86-64 function through
// that path is not currently supported — extending `compile_raw` to
// dispatch on the input function's architecture (via a new
// `X86ISelFunction` input variant) is a JIT API change tracked separately
// and is explicitly out of scope for this smoke test (see task prompt).
//
// For this smoke we bypass the `JitCompiler` wrapper and exercise the
// raw x86-64 compile-and-execute path directly:
//   1. Build an `X86ISelFunction` (reuses the existing
//      `build_x86_const_test_function` / `build_x86_add_test_function`
//      helpers from `llvm2_codegen::x86_64`).
//   2. Compile to raw machine code bytes via `X86Pipeline::compile_function`
//      with `X86OutputFormat::RawBytes` and `emit_frame = false` so the
//      emitted sequence is directly callable with the System V AMD64 ABI
//      (args in RDI/RSI, return in RAX) without touching RSP.
//   3. Allocate a writable page via `libc::mmap`, copy the bytes in,
//      flip to `PROT_READ | PROT_EXEC` via `libc::mprotect`, and invoke
//      the page as a function pointer.
//   4. Assert the returned value matches the expected semantic.
//
// x86-64 has coherent I/D caches, so no explicit `flush_icache` is needed
// after writing to executable memory (see `jit.rs:748-753`).
//
// # Acceptance criteria (#467)
//
// - [x] Test file `crates/llvm2-codegen/tests/jit_integration_x86_64.rs`
//       (this file).
// - [x] Test is SKIPPED (not failed) on AArch64 hosts via cfg-gate
//       (`#![cfg(target_arch = "x86_64")]`).
// - [x] No regression in AArch64 JIT tests (this file does not touch the
//       AArch64 JIT path; `jit_integration.rs` is unmodified).
// - [ ] Test runs on x86-64 hosts (Linux x86_64, macOS Intel). This
//       repository's primary CI/dev hosts are AArch64 (Apple Silicon),
//       so validation on an x86-64 host is pending a future run there.

#![cfg(target_arch = "x86_64")]

use llvm2_codegen::x86_64::{
    build_x86_add_test_function, build_x86_const_test_function, X86OutputFormat, X86Pipeline,
    X86PipelineConfig,
};

// ---------------------------------------------------------------------------
// Minimal raw mmap/mprotect/munmap bindings for test-local JIT execution.
//
// We avoid pulling in the `libc` crate as a dev-dependency — the codegen
// crate's dependency graph is intentionally lean and we don't want a test-
// only binding crate leaking into the production build even transitively.
// POSIX `mmap` / `mprotect` / `munmap` are ABI-stable and their prototypes
// have been frozen since SUSv3. Declaring them directly via `extern "C"`
// is safe and self-contained.
// ---------------------------------------------------------------------------

#[allow(non_camel_case_types)]
type c_void = core::ffi::c_void;
#[allow(non_camel_case_types)]
type c_int = core::ffi::c_int;
#[allow(non_camel_case_types)]
type size_t = usize;
#[allow(non_camel_case_types)]
type off_t = i64;

const PROT_NONE: c_int = 0;
const PROT_READ: c_int = 1;
const PROT_WRITE: c_int = 2;
const PROT_EXEC: c_int = 4;

const MAP_PRIVATE: c_int = 0x0002;
#[cfg(target_os = "macos")]
const MAP_ANON: c_int = 0x1000;
#[cfg(target_os = "linux")]
const MAP_ANON: c_int = 0x0020;

const MAP_FAILED: *mut c_void = !0usize as *mut c_void;

unsafe extern "C" {
    fn mmap(
        addr: *mut c_void,
        len: size_t,
        prot: c_int,
        flags: c_int,
        fd: c_int,
        offset: off_t,
    ) -> *mut c_void;
    fn mprotect(addr: *mut c_void, len: size_t, prot: c_int) -> c_int;
    fn munmap(addr: *mut c_void, len: size_t) -> c_int;
}

const PAGE_SIZE: usize = 4096; // x86-64 on macOS + Linux

fn page_align(len: usize) -> usize {
    (len + PAGE_SIZE - 1) & !(PAGE_SIZE - 1)
}

/// RAII-style executable code buffer. Allocates a page, writes the code,
/// flips to RX. `Drop` releases the mapping.
struct ExecPage {
    ptr: *mut c_void,
    size: usize,
}

impl ExecPage {
    fn new(code: &[u8]) -> Self {
        assert!(!code.is_empty(), "code must be nonempty");
        let size = page_align(code.len());
        // SAFETY: standard POSIX mmap anonymous allocation; we check
        // MAP_FAILED below before dereferencing.
        let ptr = unsafe {
            mmap(
                core::ptr::null_mut(),
                size,
                PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANON,
                -1,
                0,
            )
        };
        assert!(ptr != MAP_FAILED, "mmap failed");
        let _ = PROT_NONE; // silence unused warning on non-debug builds

        // SAFETY: `ptr` is a page-aligned writable region of at least
        // `size >= code.len()` bytes; `code.as_ptr()` is valid for
        // `code.len()` bytes; regions do not overlap.
        unsafe {
            core::ptr::copy_nonoverlapping(code.as_ptr(), ptr as *mut u8, code.len());
        }

        // Flip the page to RX. x86-64 has coherent I/D caches, no icache
        // flush needed (see jit.rs sys::flush_icache x86_64 branch).
        // SAFETY: `ptr`/`size` describe the live mapping just produced.
        let rc = unsafe { mprotect(ptr, size, PROT_READ | PROT_EXEC) };
        assert_eq!(rc, 0, "mprotect RX failed");

        ExecPage { ptr, size }
    }

    fn as_ptr(&self) -> *const u8 {
        self.ptr as *const u8
    }
}

impl Drop for ExecPage {
    fn drop(&mut self) {
        // SAFETY: `ptr`/`size` describe the live mapping created in `new`.
        // After `drop` the `ExecPage` is gone and no reference survives.
        unsafe {
            let _ = munmap(self.ptr, self.size);
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: compile an X86ISelFunction to raw callable bytes.
//
// `emit_frame = false` is required here — the test invokes the returned
// bytes directly as an `extern "C"` leaf function. With a prologue that
// pushes RBP/saves RSP, the simplified regalloc + test builders (which do
// not allocate stack slots) would still need epilogue RBP pop; the leaf
// form (no frame) is the smallest self-contained executable sequence.
// ---------------------------------------------------------------------------

fn compile_leaf(func: &llvm2_lower::x86_64_isel::X86ISelFunction) -> Vec<u8> {
    let pipeline = X86Pipeline::new(X86PipelineConfig {
        emit_frame: false,
        output_format: X86OutputFormat::RawBytes,
        ..X86PipelineConfig::default()
    });
    pipeline
        .compile_function(func)
        .expect("x86-64 compile_function should succeed for the smoke test")
}

// ---------------------------------------------------------------------------
// Test: return constant — the simplest JIT-executable smoke.
//
// Compiles `fn const42() -> i64 { 42 }`, mmap+mprotect-RX, invokes it,
// asserts the return value.
// ---------------------------------------------------------------------------

#[test]
fn test_x86_64_jit_const42() {
    let func = build_x86_const_test_function();
    let code = compile_leaf(&func);

    assert!(
        !code.is_empty(),
        "x86-64 compile_function should produce nonempty code bytes"
    );

    let page = ExecPage::new(&code);

    // SAFETY: `page.as_ptr()` points at an RX page containing an x86-64
    // leaf function matching the System V AMD64 ABI `extern "C" fn() -> i64`
    // signature produced by `build_x86_const_test_function`. The `ExecPage`
    // outlives the call.
    let f: extern "C" fn() -> i64 = unsafe { core::mem::transmute(page.as_ptr()) };

    assert_eq!(f(), 42);
}

// ---------------------------------------------------------------------------
// Test: two-argument add — exercises the two-address fixup path.
//
// Compiles `fn add(a: i64, b: i64) -> i64 { a + b }`. The ISel emits
// three-address `ADD v2, v0, v1` and the pipeline's `fixup_two_address`
// pass (between regalloc and prologue/epilogue) inserts a `MOV` to match
// x86-64's two-address `ADD dst, src` form. Without that pass the
// returned value would be wrong — this smoke test therefore also
// indirectly exercises #305's fix.
// ---------------------------------------------------------------------------

#[test]
fn test_x86_64_jit_add() {
    let func = build_x86_add_test_function();
    let code = compile_leaf(&func);

    assert!(!code.is_empty(), "x86-64 add should compile to nonempty bytes");

    let page = ExecPage::new(&code);

    // SAFETY: see `test_x86_64_jit_const42`. System V AMD64 passes first
    // two i64 args in RDI/RSI; `build_x86_add_test_function` is built
    // against that ABI.
    let f: extern "C" fn(i64, i64) -> i64 = unsafe { core::mem::transmute(page.as_ptr()) };

    assert_eq!(f(3, 4), 7);
    assert_eq!(f(0, 0), 0);
    assert_eq!(f(-1, 1), 0);
    assert_eq!(f(100, 200), 300);
    assert_eq!(f(i64::MAX, 0), i64::MAX);
    assert_eq!(f(-100, -200), -300);
}
