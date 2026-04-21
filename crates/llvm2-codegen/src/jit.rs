// llvm2-codegen/jit.rs - In-memory JIT execution via raw syscalls
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0

//! In-memory JIT compilation for LLVM2.
//!
//! Compiles IR functions directly to callable function pointers in memory,
//! bypassing Mach-O emission, the system linker, and all disk I/O.
//! Uses raw syscalls for executable memory management, plus `dlsym` for
//! process-visible symbol fallback.
//!
//! # Pipeline
//!
//! Reuses existing phases 1-8 (ISel through AArch64 encoding), then:
//! 1. Lay out all functions contiguously in a buffer
//! 2. Resolve internal fixups (cross-function BL/B calls)
//! 3. Resolve external symbols via veneer trampolines
//! 4. Allocate executable memory, copy code, flush icache
//! 5. Return [`ExecutableBuffer`] with symbol→offset map
//!
//! # Thread-Local Storage (TLS)
//!
//! The JIT does not emit TLS intrinsics. Callers resolve thread-local
//! addresses in Rust and pass them to JIT-compiled code as pointer-typed
//! `extern "C"` arguments, and the JIT treats them as ordinary pointers.
//!
//! Safety invariants:
//! - The resolved pointer is only valid on the thread that resolved it.
//! - The JIT-compiled callee must be invoked on the same thread that resolved
//!   the address.
//! - If the thread exits, the pointer is dangling.
//! - Passing the pointer across threads is UB.
//!
//! Crates that need to surface a TLS pointer through a callback can use
//! `UnsafeCell<T>` with `.with()` to get a stable address for the closure's
//! duration. That address does not outlive the closure invocation's thread.
//!
//! ```rust
//! use std::cell::UnsafeCell;
//!
//! std::thread_local! {
//!     static SCRATCH: UnsafeCell<u64> = UnsafeCell::new(0);
//! }
//!
//! fn call_jit(f: extern "C" fn(*mut u8)) {
//!     SCRATCH.with(|cell| {
//!         let ptr = cell.get() as *mut u8; // valid on THIS thread only
//!         f(ptr);
//!     });
//! }
//! ```
//!
//! Option B, emitting `mrs TPIDR_EL0` for `#[thread_local]` statics, is a
//! future optimization tracked separately.
//!
//! # Supported hosts
//!
//! The JIT execution path (`mmap` + `mprotect` + `dlsym`) is implemented for
//! four `(target_arch, target_os)` combinations:
//!
//! | target_arch | target_os | MAP_JIT | Page size | Status |
//! |---|---|---|---|---|
//! | `aarch64`   | `macos`   | yes     | 16 KiB    | primary development host |
//! | `aarch64`   | `linux`   | no      | 4 KiB     | supported (#346) |
//! | `x86_64`    | `macos`   | no      | 4 KiB     | supported |
//! | `x86_64`    | `linux`   | no      | 4 KiB     | supported (#346) |
//!
//! Any other host fails to compile with a `compile_error!` in `sys::platform`
//! so the mismatch is obvious. Running-on-Linux prerequisites: a kernel that
//! permits `mprotect(..., PROT_EXEC)` on anonymous mappings (any mainstream
//! distribution kernel qualifies) and a libc providing `dlsym` (glibc, musl,
//! or equivalent). No Mach-O / ELF object-file machinery is used by this
//! path — code is written into an anonymous mmap and invoked directly.
//!
//! Linux CI is not wired up in this repository because repo policy
//! (`design doc`) forbids `.github/workflows/`; manual smoke testing on a
//! Linux host is the intended verification route.
//!
//! # Calling convention for JIT-compiled symbols (stable contract)
//!
//! **External callers (z4, tla2, …) may `std::mem::transmute` a symbol
//! pointer returned by [`ExecutableBuffer::get_fn_ptr_bound`] /
//! [`ExecutableBuffer::get_fn_bound`] into an `extern "C" fn(...)` pointer
//! whose signature matches the tMIR function type, and invoke it directly
//! from Rust.** No wrapper, trampoline, stack-alignment shim, or
//! shadow-stack management is required beyond what the compiled function
//! itself already performs in its prologue/epilogue.
//!
//! This is a P0 compatibility contract. Any silent change to this contract
//! (e.g. a future optimization that allocates across callee-saved registers
//! without saving them, or that deviates from the host ABI's argument /
//! return conventions) **must** be gated behind an opt-in and documented
//! here. Breaking it silently is a P0 regression.
//!
//! ## AArch64 (macOS and Linux): Apple DarwinPCS / AAPCS64
//!
//! Source of truth: [`llvm2_lower::abi::AppleAArch64ABI`], with register
//! definitions in [`llvm2_ir::aarch64_regs`] and
//! [`llvm2_ir::regs`]. We implement Apple DarwinPCS, which is the
//! AAPCS64 base with Apple deltas (X18 reserved, frame pointer mandatory,
//! variadic arguments always on the stack, 16-byte SP alignment).
//!
//! Standard Linux aarch64 targets the same AAPCS64 base but with a
//! register-and-stack `va_list` layout. The fixed-argument contract
//! below is identical on both OSes, so `extern "C" fn` transmute is
//! sound on either host for non-variadic signatures. Variadic signatures
//! work today on Apple only (see "Deviations & gaps" below).
//!
//! - Integer / pointer / `bool` arguments:  `X0..=X7`, then stack.
//! - `i128` arguments:                      consecutive GPR pair
//!                                          `(X0,X1)`, `(X2,X3)`, …;
//!                                          overflow is 16-byte aligned
//!                                          on the stack.
//! - Floating-point `f32`/`f64` arguments:  `V0..=V7` (32-bit view
//!                                          `S0..=S7`, 64-bit view
//!                                          `D0..=D7`), then stack.
//! - `v128` / NEON arguments:               `V0..=V7`, then 16-byte
//!                                          aligned stack slots.
//! - Homogeneous Floating-point Aggregate
//!   (1–4 same-type `f32`/`f64` fields):    consecutive typed FPR
//!                                          sequence `S0..=S7` (F32 HFA)
//!                                          or `D0..=D7` (F64 HFA); all-
//!                                          or-nothing, falls back to the
//!                                          stack if the whole HFA cannot
//!                                          be placed.
//! - Small aggregate (≤ 8 bytes):           one GPR from `X0..=X7`.
//! - Medium aggregate (9–16 bytes):         caller places the value in
//!                                          memory and passes a pointer
//!                                          in the next free `X`.
//! - Large aggregate (> 16 bytes):          caller passes a pointer in
//!                                          the next free `X` (indirect).
//!
//! - Integer / pointer return:              `X0` (single), `(X0, X1)` for
//!                                          `i128`.
//! - FP / `v128` return:                    `V0`.
//! - HFA return (1–4 FP fields):            `S0..=S3` (F32) or
//!                                          `D0..=D3` (F64).
//! - Small aggregate return (≤ 8 bytes):    `X0`.
//! - Medium aggregate return (9–16 bytes):  `X0` + `X1` (record-pair).
//! - Struct return > 16 bytes (sret):       caller allocates the return
//!                                          buffer and passes its pointer
//!                                          in **`X8`** (not a hidden
//!                                          first `X0` argument). This
//!                                          matches AAPCS64 §6.9.
//!
//! - Callee-saved GPRs: `X19..=X28`, plus frame pointer `X29` (FP) and
//!   link register `X30` (LR). `X18` is reserved on Apple (platform
//!   register) and treated as call-clobbered.
//! - Callee-saved FPRs: `V8..=V15` (lower 64 bits only per AAPCS64; the
//!   upper half is call-clobbered).
//! - Call-clobbered GPRs: `X0..=X18`.
//! - Call-clobbered FPRs: `V0..=V7`, `V16..=V31`.
//! - Stack pointer: 16-byte aligned at every public entry and at every
//!   call site; enforced by the prologue (see `llvm2-codegen/src/frame.rs`).
//!
//! ## x86-64 (Linux and macOS): System V AMD64 ABI
//!
//! Source of truth: [`llvm2_lower::x86_64_isel`] (formal-argument,
//! call, and return lowering), with register definitions in
//! [`llvm2_ir::x86_64_regs`]. Windows x64 is **not** a supported JIT
//! host (see the host table above).
//!
//! - Integer / pointer arguments:           `RDI, RSI, RDX, RCX, R8, R9`,
//!                                          then stack at `[RBP+16]`,
//!                                          `[RBP+24]`, …
//! - FP `f32`/`f64` arguments:              `XMM0..=XMM7`, then stack.
//! - Variadic: `AL` holds the count of XMM argument registers used
//!   (0–8) at the call site, matching the System V requirement.
//!
//! - Integer / pointer return:              `RAX` (and `RDX` for the
//!                                          second integer return slot).
//! - FP return:                             `XMM0` (and `XMM1` for the
//!                                          second FP return slot).
//!
//! - Callee-saved GPRs: `RBX, RBP, R12, R13, R14, R15`.
//! - Caller-saved (clobbered) GPRs: `RAX, RCX, RDX, RSI, RDI, R8, R9,
//!   R10, R11` (plus the implicit RSP/RIP changes from CALL/RET).
//! - All XMM registers (`XMM0..=XMM15`) are caller-saved on System V
//!   (unlike Windows x64 where `XMM6..=XMM15` are callee-saved).
//! - Stack: 16-byte aligned at every call boundary.
//!
//! Aggregate-return handling follows System V §3.2.3. Today the JIT
//! front-door tests exercise scalar and small-aggregate returns; see
//! "Deviations & gaps" for the current state of large / sret returns
//! on x86-64.
//!
//! ## What "transmute is sound" means here
//!
//! A JIT caller writes:
//!
//! ```ignore
//! let buf: llvm2_codegen::ExecutableBuffer = /* from JitCompiler */;
//! let f: extern "C" fn(i64, i64) -> i64 =
//!     unsafe { buf.get_fn("add").expect("add") };
//! assert_eq!(f(3, 4), 7);
//! ```
//!
//! This is sound **iff** the tMIR signature of `"add"` is `(i64, i64) -> i64`
//! and the host is one of the four supported `(arch, os)` tuples. The
//! test suite in `crates/llvm2-codegen/tests/jit_integration.rs` already
//! uses this exact pattern for over 20 integration tests (search for
//! `extern "C" fn` in that file); those tests act as the regression
//! guard on this contract. Any codegen change that breaks them is a
//! breaking ABI change.
//!
//! The `get_fn` / `get_fn_bound` APIs `assert_eq!` that `size_of::<F>()
//! == size_of::<*const u8>()`, which gives a runtime check that `F` is
//! a pointer-sized type (i.e. a single function pointer, not a closure
//! or a wide pointer). The caller is still responsible for signature
//! compatibility.
//!
//! ## Deviations & gaps from the host C ABI
//!
//! As of today (2026-04-19) the only known gaps are in partial feature
//! coverage, **not** in ABI deviation. For every signature the backend
//! actually accepts, the emitted code matches the host C ABI.
//!
//! - **Windows x64 is not supported.** Attempting to build the crate on
//!   a Windows host fails at compile time via `compile_error!` in
//!   `sys::platform`. No silent ABI mismatch is possible.
//! - **Variadic functions on non-Apple aarch64:** the `va_list` lowering
//!   in `llvm2-lower/src/va_list.rs` implements the Apple DarwinPCS
//!   shape (`va_list` is a plain `char*` into the stack argument area),
//!   not the Linux AAPCS64 five-field struct. Non-variadic signatures
//!   are unaffected and remain ABI-compatible on Linux aarch64.
//! - **SIMD / `v128` return values** use `V0` (aarch64) / `XMM0` (x86-64)
//!   as a full vector register, not split across integer registers. This
//!   matches both host ABIs.
//! - **Struct return > 16 bytes (aarch64):** uses `X8` (sret), not a
//!   hidden first `X0` argument. This matches AAPCS64 §6.9.
//! - **x86-64 large / sret aggregate returns:** the front-door JIT tests
//!   exercise scalar and ≤16-byte aggregate returns. Large-struct sret
//!   on x86-64 follows System V §3.2.3 by construction in
//!   `llvm2-lower/src/x86_64_isel.rs`; callers that need it should
//!   add a regression test and are welcome to tighten this doc.
//!
//! If you find a real deviation from the host C ABI while reading the
//! lowering or codegen code, that is a P0 bug — file an issue and
//! update this section in the same commit.
//!
//! # Profile counter & timing-cell lifetime (issues #478, #364, #494)
//!
//! When the JIT is configured with any
//! [`ProfileHookMode`] above `None` — specifically
//! [`ProfileHookMode::CallCounts`], [`ProfileHookMode::CallCountsAndTiming`],
//! [`ProfileHookMode::BlockCounts`] or
//! [`ProfileHookMode::BlockCountsAndTiming`] — the codegen phase emits
//! trampolines whose literal pools hold **raw `*const AtomicU64` pointers**
//! into per-function or per-block counter cells (and, under
//! `BlockCountsAndTiming`, a single buffer-wide `*mut TimingState`). The
//! pointees are `Box<AtomicU64>` / `Box<BlockTimingCell>` /
//! `Box<TimingState>` allocations owned by the [`ExecutableBuffer`].
//!
//! The lifetime invariant is:
//!
//! > **Every counter allocation, every timing-cell allocation, and the
//! > single timing state allocation must outlive the executable mapping
//! > they are referenced from.**
//!
//! This is guaranteed structurally, not by convention, because:
//!
//! 1. The `Box` allocations live in
//!    [`ExecutableBuffer::counters`] / `timing_cells` /
//!    `timing_state` — fields of the same buffer that owns the mmap'd
//!    `memory`.
//! 2. `Box<T>` pins its heap address: a counter's address is fixed from
//!    allocation to drop regardless of `HashMap` resize, insertion, or
//!    iteration. The trampoline's literal pool can therefore cache the
//!    address at compile time and read / write it indefinitely.
//! 3. `impl Drop for ExecutableBuffer` runs in the order:
//!    `munmap(memory)` first (user `Drop::drop`), then Rust-driven field
//!    drops in declaration order. Because `memory` is dropped before
//!    `counters` / `timing_cells` / `timing_state`, the executable code —
//!    which is the **only** holder of raw counter pointers — is unmapped
//!    strictly before any counter Box is dropped. There is therefore no
//!    window in which the trampoline can execute against a freed counter.
//!
//! The hazardous case is **not** counter-lifetime vs buffer-lifetime
//! (structural, as above) but **buffer-lifetime vs in-flight JIT call**:
//! if a caller drops the [`ExecutableBuffer`] while another thread is
//! inside a JIT-compiled function, the `munmap` invalidates the text
//! pages (and the embedded counter pointers) mid-call, producing the
//! same use-after-free hazard described for
//! [`ExecutableBuffer::get_fn_ptr`] / [`ExecutableBuffer::get_fn`]. The
//! mitigation is identical: prefer the lifetime-bound
//! [`ExecutableBuffer::get_fn_bound`] / [`ExecutableBuffer::get_fn_ptr_bound`]
//! APIs, or wrap the buffer in `Arc<ExecutableBuffer>` and only drop it
//! after every call has returned. See the `SAFETY` note on
//! [`ExecutableBuffer`]'s `Send`/`Sync` impls and issue #355 for the
//! legacy-API hazard, and #478 / #364 / #494 for the counter extensions.
//!
//! ## Cross-references
//!
//! - [`llvm2_lower::abi::AppleAArch64ABI`] — aarch64 parameter/return classifier.
//! - [`llvm2_lower::x86_64_isel`] — x86-64 formal arg / call / return lowering.
//! - `crates/llvm2-codegen/src/frame.rs` — aarch64 prologue/epilogue,
//!   SP alignment, callee-saved area.
//! - `crates/llvm2-codegen/tests/jit_integration.rs` — live regression
//!   tests that exercise `transmute` to `extern "C" fn` for scalar,
//!   aggregate, call-through-host, and host-callback signatures.
//! - Sibling instruction / pattern work: `#429` (SMULH opcode + isel)
//!   and `#430` (recognize ADDS+B.VS idiom from tMIR i128 overflow).
//! - Upstream tracker: `#431`.

use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use thiserror::Error;

use crate::macho::fixup::{Fixup, FixupTarget};
use crate::pipeline::{
    DispatchVerifyMode, OptLevel, Pipeline, PipelineConfig, PipelineError,
    encode_function_with_fixups, encode_function_with_fixups_and_blocks,
};
use llvm2_ir::function::MachFunction as IrMachFunction;
use llvm2_ir::types::BlockId;

// ---------------------------------------------------------------------------
// Raw syscall interface for executable memory management.
// ---------------------------------------------------------------------------
mod sys {
    const PROT_READ: i32 = 1;
    const PROT_WRITE: i32 = 2;
    const PROT_EXEC: i32 = 4;
    pub const RW: i32 = PROT_READ | PROT_WRITE;
    pub const RX: i32 = PROT_READ | PROT_EXEC;

    // -- AArch64 macOS (Apple Silicon) -----------------------------------------
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    mod platform {
        // macOS Mach syscall numbers (via XNU sys/syscall.h)
        pub const SYS_MMAP: u64 = 197;
        pub const SYS_MUNMAP: u64 = 73;
        pub const SYS_MPROTECT: u64 = 74;
        // MAP_PRIVATE | MAP_ANONYMOUS | MAP_JIT
        pub const MAP_FLAGS: i32 = 0x0002 | 0x1000 | 0x0800;
        pub const PAGE_SIZE: usize = 16384; // Apple Silicon
    }

    // -- AArch64 Linux ---------------------------------------------------------
    #[cfg(all(target_arch = "aarch64", target_os = "linux"))]
    mod platform {
        // Linux AArch64 syscall numbers
        pub const SYS_MMAP: u64 = 222;
        pub const SYS_MUNMAP: u64 = 215;
        pub const SYS_MPROTECT: u64 = 226;
        pub const MAP_FLAGS: i32 = 0x0002 | 0x0020; // MAP_PRIVATE | MAP_ANONYMOUS
        pub const PAGE_SIZE: usize = 4096;
    }

    // -- x86-64 macOS ----------------------------------------------------------
    #[cfg(all(target_arch = "x86_64", target_os = "macos"))]
    mod platform {
        // macOS x86-64 syscall numbers use 0x2000000 prefix (BSD class).
        // Reference: XNU bsd/kern/syscalls.master
        pub const SYS_MMAP: u64 = 0x2000000 + 197;
        pub const SYS_MUNMAP: u64 = 0x2000000 + 73;
        pub const SYS_MPROTECT: u64 = 0x2000000 + 74;
        // MAP_PRIVATE | MAP_ANONYMOUS (no MAP_JIT needed on x86-64)
        pub const MAP_FLAGS: i32 = 0x0002 | 0x1000;
        pub const PAGE_SIZE: usize = 4096; // x86-64 uses 4K pages
    }

    // -- x86-64 Linux ----------------------------------------------------------
    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
    mod platform {
        // Linux x86-64 syscall numbers (via asm/unistd_64.h)
        pub const SYS_MMAP: u64 = 9;
        pub const SYS_MUNMAP: u64 = 11;
        pub const SYS_MPROTECT: u64 = 10;
        pub const MAP_FLAGS: i32 = 0x0002 | 0x0020; // MAP_PRIVATE | MAP_ANONYMOUS
        pub const PAGE_SIZE: usize = 4096;
    }

    // -- Unsupported (arch, os) ------------------------------------------------
    // Give a readable compile-time error instead of a cryptic "cannot find
    // `SYS_MMAP`" from `pub use platform::*;` below. Supported hosts are
    // listed in the module-level doc comment. Issue #346 tracks Linux
    // support; additional targets require new syscall and MAP_FLAGS
    // constants. (Keep this `any()` list in sync with the four `cfg(all(...))`
    // `platform` modules above.)
    #[cfg(not(any(
        all(target_arch = "aarch64", target_os = "macos"),
        all(target_arch = "aarch64", target_os = "linux"),
        all(target_arch = "x86_64", target_os = "macos"),
        all(target_arch = "x86_64", target_os = "linux"),
    )))]
    compile_error!(
        "llvm2-codegen JIT supports aarch64/x86_64 on macos/linux only; \
         see crate-level docs in src/jit.rs for the supported-hosts table."
    );

    pub use platform::*;

    // -- Syscall result type ---------------------------------------------------
    // macOS/XNU signals syscall errors via the carry flag (CPSR.C on AArch64,
    // CF on x86-64), NOT via a negative return value (that's the Linux
    // convention). When carry is set, x0/rax contains the positive errno.
    // We capture carry into `err` so callers can correctly distinguish
    // success from failure for all syscalls (not just mmap).
    struct SyscallResult {
        val: i64,
        /// 1 if carry flag was set (error), 0 otherwise.
        /// On Linux this is always 0 — we use the traditional negative-return
        /// check in `check_error`, so `err` is read only on macOS. The
        /// `#[allow(dead_code)]` is conditional on the Linux build where
        /// the field is dead; on macOS it is actively consumed and the
        /// allow has no effect. Keeping the field shape identical across
        /// platforms lets every syscall wrapper construct the same struct
        /// without cfg-gating every call site. (Issue #346.)
        #[cfg_attr(target_os = "linux", allow(dead_code))]
        err: u64,
    }

    // -- AArch64 macOS raw syscall wrappers ------------------------------------
    // AArch64 ABI: syscall number in x16, args in x0-x5, return in x0.
    // Error convention: carry flag (CPSR.C) set on error, x0 = positive errno.
    // Instruction: svc #0x80, then cset to capture carry.

    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    unsafe fn syscall6(
        nr: u64,
        a0: u64,
        a1: u64,
        a2: u64,
        a3: u64,
        a4: u64,
        a5: u64,
    ) -> SyscallResult {
        let val: i64;
        let err: u64;
        // SAFETY (caller-upheld): `unsafe fn` contract delegates raw-syscall
        // responsibility (valid fd/addr/prot args) to the caller; the inline
        // `svc #0x80` itself only uses registers and sets the carry flag.
        unsafe {
            core::arch::asm!(
                "svc #0x80",
                "cset {err}, cs",
                err = out(reg) err,
                in("x16") nr,
                inout("x0") a0 => val,
                in("x1") a1,
                in("x2") a2,
                in("x3") a3,
                in("x4") a4,
                in("x5") a5,
                options(nostack),
            );
        }
        SyscallResult { val, err }
    }

    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    unsafe fn syscall3(nr: u64, a0: u64, a1: u64, a2: u64) -> SyscallResult {
        let val: i64;
        let err: u64;
        // SAFETY: see `syscall6` above — same caller contract applies.
        unsafe {
            core::arch::asm!(
                "svc #0x80",
                "cset {err}, cs",
                err = out(reg) err,
                in("x16") nr,
                inout("x0") a0 => val,
                in("x1") a1,
                in("x2") a2,
                options(nostack),
            );
        }
        SyscallResult { val, err }
    }

    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    unsafe fn syscall2(nr: u64, a0: u64, a1: u64) -> SyscallResult {
        let val: i64;
        let err: u64;
        // SAFETY: see `syscall6` above — same caller contract applies.
        unsafe {
            core::arch::asm!(
                "svc #0x80",
                "cset {err}, cs",
                err = out(reg) err,
                in("x16") nr,
                inout("x0") a0 => val,
                in("x1") a1,
                options(nostack),
            );
        }
        SyscallResult { val, err }
    }

    // -- AArch64 Linux raw syscall wrappers ------------------------------------
    // Linux AArch64: negative return = -errno. No carry flag convention.

    #[cfg(all(target_arch = "aarch64", target_os = "linux"))]
    unsafe fn syscall6(
        nr: u64,
        a0: u64,
        a1: u64,
        a2: u64,
        a3: u64,
        a4: u64,
        a5: u64,
    ) -> SyscallResult {
        let val: i64;
        // SAFETY: see macOS `syscall6` — same caller contract applies.
        unsafe {
            core::arch::asm!(
                "svc #0",
                in("x8") nr,
                inout("x0") a0 => val,
                in("x1") a1,
                in("x2") a2,
                in("x3") a3,
                in("x4") a4,
                in("x5") a5,
                options(nostack),
            );
        }
        SyscallResult { val, err: 0 }
    }

    #[cfg(all(target_arch = "aarch64", target_os = "linux"))]
    unsafe fn syscall3(nr: u64, a0: u64, a1: u64, a2: u64) -> SyscallResult {
        let val: i64;
        // SAFETY: see macOS `syscall6` — same caller contract applies.
        unsafe {
            core::arch::asm!(
                "svc #0",
                in("x8") nr,
                inout("x0") a0 => val,
                in("x1") a1,
                in("x2") a2,
                options(nostack),
            );
        }
        SyscallResult { val, err: 0 }
    }

    #[cfg(all(target_arch = "aarch64", target_os = "linux"))]
    unsafe fn syscall2(nr: u64, a0: u64, a1: u64) -> SyscallResult {
        let val: i64;
        // SAFETY: see macOS `syscall6` — same caller contract applies.
        unsafe {
            core::arch::asm!(
                "svc #0",
                in("x8") nr,
                inout("x0") a0 => val,
                in("x1") a1,
                options(nostack),
            );
        }
        SyscallResult { val, err: 0 }
    }

    // -- x86-64 macOS raw syscall wrappers -------------------------------------
    // x86-64 macOS: syscall number in rax, args in rdi/rsi/rdx/r10/r8/r9.
    // Return in rax. Clobbers rcx and r11.
    // Error convention: carry flag (CF) set on error, rax = positive errno.

    #[cfg(all(target_arch = "x86_64", target_os = "macos"))]
    unsafe fn syscall6(
        nr: u64,
        a0: u64,
        a1: u64,
        a2: u64,
        a3: u64,
        a4: u64,
        a5: u64,
    ) -> SyscallResult {
        let val: i64;
        let err: u64;
        // SAFETY: see aarch64 macOS `syscall6` — same caller contract applies.
        unsafe {
            core::arch::asm!(
                "syscall",
                "setc {err_byte}",
                "movzx {err}, {err_byte}",
                err_byte = out(reg_byte) _,
                err = out(reg) err,
                inout("rax") nr as i64 => val,
                in("rdi") a0,
                in("rsi") a1,
                in("rdx") a2,
                in("r10") a3,
                in("r8") a4,
                in("r9") a5,
                out("rcx") _,
                out("r11") _,
                options(nostack),
            );
        }
        SyscallResult { val, err }
    }

    #[cfg(all(target_arch = "x86_64", target_os = "macos"))]
    unsafe fn syscall3(nr: u64, a0: u64, a1: u64, a2: u64) -> SyscallResult {
        let val: i64;
        let err: u64;
        // SAFETY: see aarch64 macOS `syscall6` — same caller contract applies.
        unsafe {
            core::arch::asm!(
                "syscall",
                "setc {err_byte}",
                "movzx {err}, {err_byte}",
                err_byte = out(reg_byte) _,
                err = out(reg) err,
                inout("rax") nr as i64 => val,
                in("rdi") a0,
                in("rsi") a1,
                in("rdx") a2,
                out("rcx") _,
                out("r11") _,
                options(nostack),
            );
        }
        SyscallResult { val, err }
    }

    #[cfg(all(target_arch = "x86_64", target_os = "macos"))]
    unsafe fn syscall2(nr: u64, a0: u64, a1: u64) -> SyscallResult {
        let val: i64;
        let err: u64;
        // SAFETY: see aarch64 macOS `syscall6` — same caller contract applies.
        unsafe {
            core::arch::asm!(
                "syscall",
                "setc {err_byte}",
                "movzx {err}, {err_byte}",
                err_byte = out(reg_byte) _,
                err = out(reg) err,
                inout("rax") nr as i64 => val,
                in("rdi") a0,
                in("rsi") a1,
                out("rcx") _,
                out("r11") _,
                options(nostack),
            );
        }
        SyscallResult { val, err }
    }

    // -- x86-64 Linux raw syscall wrappers -------------------------------------
    // Linux x86-64: negative return = -errno. No carry flag convention.

    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
    unsafe fn syscall6(
        nr: u64,
        a0: u64,
        a1: u64,
        a2: u64,
        a3: u64,
        a4: u64,
        a5: u64,
    ) -> SyscallResult {
        let val: i64;
        // SAFETY: see aarch64 macOS `syscall6` — same caller contract applies.
        unsafe {
            core::arch::asm!(
                "syscall",
                inout("rax") nr as i64 => val,
                in("rdi") a0,
                in("rsi") a1,
                in("rdx") a2,
                in("r10") a3,
                in("r8") a4,
                in("r9") a5,
                out("rcx") _,
                out("r11") _,
                options(nostack),
            );
        }
        SyscallResult { val, err: 0 }
    }

    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
    unsafe fn syscall3(nr: u64, a0: u64, a1: u64, a2: u64) -> SyscallResult {
        let val: i64;
        // SAFETY: see aarch64 macOS `syscall6` — same caller contract applies.
        unsafe {
            core::arch::asm!(
                "syscall",
                inout("rax") nr as i64 => val,
                in("rdi") a0,
                in("rsi") a1,
                in("rdx") a2,
                out("rcx") _,
                out("r11") _,
                options(nostack),
            );
        }
        SyscallResult { val, err: 0 }
    }

    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
    unsafe fn syscall2(nr: u64, a0: u64, a1: u64) -> SyscallResult {
        let val: i64;
        // SAFETY: see aarch64 macOS `syscall6` — same caller contract applies.
        unsafe {
            core::arch::asm!(
                "syscall",
                inout("rax") nr as i64 => val,
                in("rdi") a0,
                in("rsi") a1,
                out("rcx") _,
                out("r11") _,
                options(nostack),
            );
        }
        SyscallResult { val, err: 0 }
    }

    /// Check a syscall result for error.
    ///
    /// On macOS, errors are signaled via carry flag (captured in `res.err`).
    /// When carry is set, `res.val` contains the positive errno.
    ///
    /// On Linux, errors are signaled via negative return value (-errno).
    /// `res.err` is always 0 on Linux.
    fn check_error(res: &SyscallResult) -> Option<std::io::Error> {
        #[cfg(target_os = "macos")]
        {
            if res.err != 0 {
                // macOS: carry set, val is the positive errno
                return Some(std::io::Error::from_raw_os_error(res.val as i32));
            }
        }
        #[cfg(target_os = "linux")]
        {
            if res.val < 0 {
                // Linux: negative return = -errno
                return Some(std::io::Error::from_raw_os_error((-res.val) as i32));
            }
        }
        None
    }

    pub unsafe fn mmap(len: usize, prot: i32) -> Result<*mut u8, std::io::Error> {
        // SAFETY: this `unsafe fn` wraps the raw mmap syscall; caller is
        // responsible for using the returned page correctly (see `ExecutableBuffer`).
        let res = unsafe {
            syscall6(
                SYS_MMAP,
                0, // addr: NULL (kernel chooses)
                len as u64,
                prot as u64,
                MAP_FLAGS as u64,
                u64::MAX, // fd: -1
                0,        // offset: 0
            )
        };
        if let Some(e) = check_error(&res) {
            Err(e)
        } else {
            Ok(res.val as *mut u8)
        }
    }

    pub unsafe fn munmap(addr: *mut u8, len: usize) {
        // SAFETY: caller guarantees `addr`/`len` match a prior `mmap`.
        let _ = unsafe { syscall2(SYS_MUNMAP, addr as u64, len as u64) };
    }

    pub unsafe fn mprotect(addr: *mut u8, len: usize, prot: i32) -> Result<(), std::io::Error> {
        // SAFETY: caller guarantees `addr`/`len` name a valid mapping.
        let res = unsafe { syscall3(SYS_MPROTECT, addr as u64, len as u64, prot as u64) };
        if let Some(e) = check_error(&res) {
            Err(e)
        } else {
            Ok(())
        }
    }

    pub unsafe fn flush_icache(addr: *mut u8, len: usize) {
        #[cfg(target_arch = "aarch64")]
        {
            // Walk cache lines and invalidate. Line size = 64 bytes on Apple Silicon.
            let mut p = addr as usize;
            let end = p + len;
            // SAFETY: caller supplies a valid [addr, addr+len) range for the
            // just-written executable mapping; DC/IC/DSB/ISB are side-effect
            // free beyond the cache/barrier semantics they document.
            unsafe {
                while p < end {
                    core::arch::asm!(
                        "dc cvau, {addr}",   // Clean data cache to point of unification
                        addr = in(reg) p,
                        options(nostack),
                    );
                    p += 64;
                }
                core::arch::asm!("dsb ish", options(nostack)); // Data sync barrier
                p = addr as usize;
                while p < end {
                    core::arch::asm!(
                        "ic ivau, {addr}",   // Invalidate instruction cache
                        addr = in(reg) p,
                        options(nostack),
                    );
                    p += 64;
                }
                core::arch::asm!("dsb ish", options(nostack));
                core::arch::asm!("isb", options(nostack)); // Instruction sync barrier
            }
        }
        #[cfg(target_arch = "x86_64")]
        {
            // x86-64 has coherent instruction and data caches.
            // No cache flush needed after writing code to executable memory.
            let _ = (addr, len);
        }
    }

    pub fn page_align(len: usize) -> usize {
        (len + PAGE_SIZE - 1) & !(PAGE_SIZE - 1)
    }
}

// ---------------------------------------------------------------------------
// Process symbol resolution — dlsym(RTLD_DEFAULT, name)
// ---------------------------------------------------------------------------
// `dlsym(RTLD_DEFAULT, ...)` is thread-safe per POSIX. For symbols in the main
// binary, the returned pointer is stable for the lifetime of the process.
// Callers must still ensure the resolved symbol's ABI matches the generated
// callsite ABI before invoking it.
#[cfg(unix)]
mod dl {
    use std::os::raw::{c_char, c_int, c_void};

    // RTLD_DEFAULT is an opaque pseudo-handle. Values differ per platform.
    #[cfg(target_os = "macos")]
    pub const RTLD_DEFAULT: *mut c_void = -2isize as *mut c_void;

    #[cfg(target_os = "linux")]
    pub const RTLD_DEFAULT: *mut c_void = std::ptr::null_mut();

    unsafe extern "C" {
        pub fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
        pub fn dlerror() -> *const c_char;
    }

    // Suppress unused warning where target_os is neither macos nor linux.
    #[allow(dead_code)]
    fn _unused() {
        let _: c_int = 0;
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum JitError {
    #[error("pipeline error: {0}")]
    Pipeline(#[from] PipelineError),

    #[error("unresolved symbol: {0}")]
    UnresolvedSymbol(String),

    #[error("memory allocation failed: {0}")]
    MemoryAlloc(std::io::Error),

    #[error("memory protection failed: {0}")]
    MemoryProtect(std::io::Error),

    #[error("branch out of range: offset {offset} to {target} (distance {distance})")]
    BranchOutOfRange {
        offset: u32,
        target: u64,
        distance: i64,
    },

    #[error(
        "veneer for `{symbol}` at {veneer_offset} is out of BL range from call site {offset} \
         (distance {distance}, max +-128MiB)"
    )]
    VeneerOutOfRange {
        symbol: String,
        offset: u32,
        veneer_offset: u64,
        distance: i64,
    },

    #[error("fixup offset {offset} out of bounds (code length {code_len})")]
    FixupOutOfBounds { offset: u32, code_len: usize },

    #[error(
        "duplicate JIT symbol: `{0}` (primary name or `_`-prefixed alias collides with a previously compiled function)"
    )]
    DuplicateSymbol(String),

    #[error("profile hooks are only supported on aarch64 and x86-64 hosts")]
    ProfileHooksUnsupported,

    #[error(
        "profile hook mode `{mode:?}` is a reserved #396 Phase 2 variant; \
         trampoline emission is not yet implemented. See \
         designs/2026-04-18-pgo-workflow.md and use \
         `ProfileHookMode::CallCounts` for the current Phase 1 surface."
    )]
    ProfileHookModeUnimplemented { mode: ProfileHookMode },
}

/// Maximum absolute distance for an AArch64 B/BL imm26 branch: +-128 MiB.
/// The imm26 field encodes a signed 26-bit word offset, so the reachable
/// range is [-(1 << 27), (1 << 27)) bytes from the branch instruction.
#[cfg(target_arch = "aarch64")]
const AARCH64_BRANCH26_MAX: i64 = 1 << 27;

/// Check whether an AArch64 B/BL `imm26` branch at `offset` can reach `target`.
/// Returns true if the signed distance fits in [-128 MiB, +128 MiB).
#[cfg(target_arch = "aarch64")]
fn branch26_in_range(offset: u32, target: u64) -> (bool, i64) {
    let distance = target as i64 - offset as i64;
    let in_range = distance >= -AARCH64_BRANCH26_MAX && distance < AARCH64_BRANCH26_MAX;
    (in_range, distance)
}

/// Pre-validate that every veneer trampoline is within AArch64 BL reach of
/// the call site that will patch into it.
///
/// `ext_patches` is the per-fixup list of `(bl_offset, veneer_offset, symbol)`
/// triples collected during veneer emission in
/// [`JitCompiler::compile_raw`]. This function performs the range check in
/// isolation so that (a) the check is exercised by unit tests without
/// emitting >128 MiB of real code, and (b) the validation pass in
/// `compile_raw` is a single well-named call rather than an inline loop.
///
/// On AArch64 this enforces the imm26 range `[-2^27, +2^27)` bytes.
/// On non-AArch64 hosts the check is a no-op (BL range is a property of the
/// target ISA — on macOS-x86_64 JIT hosts the veneer code would not run
/// anyway, and the BL imm26 limit does not apply).
///
/// Returns `Err(JitError::VeneerOutOfRange)` on the first unreachable pair.
/// The `_code_len` parameter is accepted (and unused) so the signature can
/// grow a bounds-check or island-aware variant later without breaking callers.
#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
fn validate_veneer_ranges(
    ext_patches: &[(u32, u64, String)],
    _code_len: usize,
) -> Result<(), JitError> {
    #[cfg(target_arch = "aarch64")]
    for (fx_off, veneer_off, sym) in ext_patches {
        let (ok, distance) = branch26_in_range(*fx_off, *veneer_off);
        if !ok {
            return Err(JitError::VeneerOutOfRange {
                symbol: sym.clone(),
                offset: *fx_off,
                veneer_offset: *veneer_off,
                distance,
            });
        }
    }
    // Suppress unused-variable warning on non-aarch64 hosts where the loop
    // is compiled out entirely.
    #[cfg(not(target_arch = "aarch64"))]
    let _ = ext_patches;
    Ok(())
}

/// Entry-hook mode for JIT-compiled functions.
///
/// # Granularity levels (#396 PGO workflow)
///
/// Variants are ordered roughly by granularity. `None` disables all hooks
/// (the zero-overhead default). `CallCounts` / `CallCountsAndTiming` are
/// the original function-entry trampolines and are the only modes fully
/// implemented at the codegen layer today.
///
/// The remaining variants (`BlockCounts`, `BlockCountsAndTiming`,
/// `EdgeCounts`, `BlockFrequency`, `LoopHeads`) are the API surface for
/// Phase 2 of #396. They are accepted by [`JitConfig`] so downstream
/// consumers (tla2 canary capture, [`llvm2_opt::pgo`]) can compile
/// against the stable API, but the trampoline emitter currently treats
/// them as unsupported and returns [`JitError::ProfileHookModeUnimplemented`]
/// from [`JitCompiler::compile_raw`]. Implementations land in follow-up
/// issues — see `designs/2026-04-18-pgo-workflow.md`.
///
/// # Public Entry-Counter API (#478)
///
/// The stable, ergonomic name for the function-entry slice is
/// [`JitConfig::emit_entry_counters`] plus
/// [`ExecutableBuffer::entry_count`],
/// [`ExecutableBuffer::reset_entry_count`], and
/// [`ExecutableBuffer::entry_counts`]. `ProfileHookMode` remains the
/// lower-level knob for callers that need to choose a specific profiling
/// mode directly.
///
/// See also [`llvm2_opt::pgo::inject`] for the MachIR-level block-counter
/// injection pass that already exists (Phase 1 landed in #396).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ProfileHookMode {
    /// No profiling hooks. Zero overhead. The default.
    None,
    /// One counter per function, incremented at the trampoline entry.
    CallCounts,
    /// `CallCounts` plus wall-time timing around the function body.
    CallCountsAndTiming,
    // ----- Phase 2 (#396) — API reserved, trampoline TODO ------------------
    /// One counter per basic block, incremented at block prologue.
    ///
    /// TODO(#396 Phase 2): requires extending
    /// `emit_profile_trampoline_aarch64` to visit every MachBlock instead
    /// of just the function entry, and allocating one [`AtomicU64`] slot
    /// per block rather than per function. The MachIR-level injection
    /// pass (`llvm2_opt::pgo::inject_block_counters`) already exists;
    /// this variant is the codegen-side hookup.
    BlockCounts,
    /// `BlockCounts` plus per-block wall-time timing.
    ///
    /// TODO(#396 Phase 2): as `BlockCounts`, plus timing probes on every
    /// block prologue and epilogue. Beware: per-block timing can dwarf
    /// the work being measured; emit only when the caller has explicitly
    /// asked for timing.
    BlockCountsAndTiming,
    /// One counter per CFG edge, incremented on the edge itself (after
    /// critical-edge splitting).
    ///
    /// TODO(#396 Phase 3): strictly more information than `BlockCounts`
    /// but doubles counter count. Requires a critical-edge-split
    /// pre-pass so each edge has a unique landing block; otherwise
    /// multi-predecessor blocks double-count. See LLVM's
    /// `llvm/lib/Transforms/IPO/PGOInstrumentation.cpp` for prior art.
    EdgeCounts,
    /// Derived per-block frequency (no new counters; computed from
    /// `BlockCounts` or `EdgeCounts` via Kirchhoff balance at profile
    /// read time).
    ///
    /// TODO(#396 Phase 2): this is the *consumer*-facing mode —
    /// requesting `BlockFrequency` from the JIT should be equivalent to
    /// requesting `BlockCounts` and having the reader derive
    /// frequencies. Added here so the `JitConfig` API matches what
    /// `ProfileUse` consumers (inline budget, unroll, block layout)
    /// ultimately want to see.
    BlockFrequency,
    /// Only loop-head blocks get counters. Captures iteration counts
    /// without per-block overhead.
    ///
    /// TODO(#396 Phase 2): implemented as a filter on top of
    /// `BlockCounts`: the instrumentation step queries the loop
    /// analysis (`llvm2_opt::loops`) and only emits counter increments
    /// at loop headers.
    LoopHeads,
}

/// Snapshot of per-function profile data exposed by [`ExecutableBuffer`].
#[derive(Debug, Clone, Copy, Default)]
pub struct ProfileStats {
    pub call_count: u64,
}

/// Returns `true` if `mode` is a granularity level for which per-function
/// counter slots must be allocated and the function-entry trampoline
/// emitted. The newer block/edge/frequency modes (#396 Phase 2) return
/// `false` here — they are accepted by the API but not yet reached from
/// `compile_raw`; [`JitCompiler::compile_raw`] rejects them with
/// [`JitError::ProfileHookModeUnimplemented`] until the trampoline work
/// lands.
fn profile_hooks_enable_counters(mode: ProfileHookMode) -> bool {
    matches!(
        mode,
        ProfileHookMode::CallCounts | ProfileHookMode::CallCountsAndTiming
    )
}

/// Returns `true` if `mode` allocates one counter per basic block and
/// emits a trampoline at the start of every block (not just the function
/// entry). Landed for [`ProfileHookMode::BlockCounts`] under #364; the
/// companion [`ProfileHookMode::BlockCountsAndTiming`] variant is handled
/// by [`profile_hooks_enable_block_timing`] and goes through a larger
/// trampoline that additionally captures a `CNTVCT_EL0` timestamp.
fn profile_hooks_enable_block_counters(mode: ProfileHookMode) -> bool {
    matches!(mode, ProfileHookMode::BlockCounts)
}

/// Returns `true` if `mode` is [`ProfileHookMode::BlockCountsAndTiming`]
/// — one `{count, total_cycles}` cell per basic block, plus a shared
/// `{prev_ts, prev_accum_ptr}` timing-state struct. The larger trampoline
/// captures `CNTVCT_EL0` on every block entry and attributes the cycles
/// between consecutive block entries to the previously-entered block.
///
/// Implemented on AArch64 only (#364 Phase 3). The x86-64 port is a
/// follow-up; on non-AArch64 hosts this mode returns
/// [`JitError::ProfileHooksUnsupported`].
fn profile_hooks_enable_block_timing(mode: ProfileHookMode) -> bool {
    matches!(mode, ProfileHookMode::BlockCountsAndTiming)
}

/// Returns `true` if `mode` is a #396 Phase 2 variant whose trampoline
/// work is still TODO. Kept in one place so the early-return in
/// `compile_raw` and the error-message test stay in sync.
fn profile_hooks_is_phase2_stub(mode: ProfileHookMode) -> bool {
    // NOTE: `BlockCountsAndTiming` is intentionally NOT in this list —
    // it is implemented by `splice_block_trampolines_with_timing_aarch64`
    // (#364 Phase 3).
    matches!(
        mode,
        ProfileHookMode::EdgeCounts
            | ProfileHookMode::BlockFrequency
            | ProfileHookMode::LoopHeads
    )
}

/// Configuration for [`JitCompiler`].
///
/// # Dispatch verification default (#375)
///
/// The `verify_dispatch` field is propagated into the underlying
/// [`PipelineConfig`] and defaults to [`DispatchVerifyMode::ErrorOnFailure`].
/// Any code path that invokes the Pipeline's dispatch verifier — for example
/// [`Pipeline::generate_and_verify_dispatch`] or
/// [`Pipeline::verify_dispatch_plan`] — will therefore return
/// [`PipelineError::DispatchVerificationFailed`] on a failing plan rather
/// than silently substituting a CPU-only fallback. LLVM2's value proposition
/// is verified codegen, so silent fallback (the previous default) is the
/// wrong behaviour by default: a verification failure must not be
/// indistinguishable from success.
///
/// Note on current reach: [`JitCompiler::compile_raw`] does not itself
/// invoke the dispatch verifier yet — the verifier is only reached via the
/// pipeline APIs mentioned above. The default still matters because callers
/// that share a [`Pipeline`] via the JIT (including future heterogeneous-
/// aware JIT entry points) inherit this policy from `JitConfig`.
///
/// Callers that explicitly want the legacy silent-fallback behaviour (for
/// example, best-effort heterogeneous dispatch on graphs where a CPU
/// fallback is always acceptable) can opt in with:
///
/// ```
/// use llvm2_codegen::{JitConfig, DispatchVerifyMode};
/// let cfg = JitConfig {
///     verify_dispatch: DispatchVerifyMode::FallbackOnFailure,
///     ..JitConfig::default()
/// };
/// ```
///
/// Set `verify_dispatch` to [`DispatchVerifyMode::Off`] to skip dispatch
/// verification entirely. Off bypasses the correctness check — prefer
/// `FallbackOnFailure` if the intent is "soft failure" rather than "no
/// check at all". The default remains `ErrorOnFailure` so failures are
/// never silently swallowed.
#[derive(Debug, Clone)]
pub struct JitConfig {
    /// Optimization level for the underlying pipeline.
    pub opt_level: OptLevel,
    /// Whether to run function-level verification after optimization.
    ///
    /// This is a separate gate from `verify_dispatch`: it controls
    /// instruction-level verification of the lowered/optimized IR, not the
    /// heterogeneous dispatch plan.
    pub verify: bool,
    /// Policy for handling dispatch-plan verification failures.
    ///
    /// Defaults to [`DispatchVerifyMode::ErrorOnFailure`] so that
    /// verification failures surface as [`JitError::Pipeline`] rather than
    /// being silently replaced by a CPU-only fallback (see #375).
    pub verify_dispatch: DispatchVerifyMode,
    /// Optional per-function profiling hooks inserted at JIT entry.
    pub profile_hooks: ProfileHookMode,
    /// Convenience flag equivalent to `profile_hooks = ProfileHookMode::CallCounts`.
    ///
    /// When `true`, the JIT emits one atomic `u64` counter per function,
    /// incremented at function entry, readable via
    /// [`ExecutableBuffer::entry_count`]. This is the public name for the
    /// function-entry slice of #364 (see issue #478).
    ///
    /// If `profile_hooks` is also set to anything other than `None`, that
    /// explicit setting wins. Use `emit_entry_counters` as the ergonomic
    /// default; reach for `profile_hooks` for finer control.
    pub emit_entry_counters: bool,
}

impl Default for JitConfig {
    fn default() -> Self {
        Self {
            opt_level: OptLevel::O2,
            verify: false,
            // #375: Default to ErrorOnFailure so dispatch-verification
            // failures are not silently swallowed. Callers that want the
            // previous silent-fallback behaviour must opt in explicitly.
            verify_dispatch: DispatchVerifyMode::ErrorOnFailure,
            profile_hooks: ProfileHookMode::None,
            emit_entry_counters: false,
        }
    }
}

/// Look up a symbol in the current process's symbol table.
///
/// Uses `dlsym(RTLD_DEFAULT, ...)` on Unix. This finds any symbol visible in
/// the current process, including `#[no_mangle] pub extern "C"` functions
/// defined in the calling binary.
///
/// Returns `None` if the symbol is not found, or if the name contains an
/// interior NUL byte (which is invalid for a C string).
///
/// # Safety
/// The returned pointer is only valid as long as the symbol exists in the
/// process. For dynamically loaded libraries this means until the library
/// is unloaded. For symbols in the main binary, the pointer is valid for
/// the lifetime of the process.
#[cfg(unix)]
pub fn lookup_process_symbol(name: &str) -> Option<*const u8> {
    let c_name = std::ffi::CString::new(name).ok()?;
    // SAFETY: Clearing dlerror state before dlsym is the documented way to
    // distinguish a NULL symbol value from a lookup failure.
    unsafe {
        dl::dlerror();
    }
    // SAFETY: `dl::RTLD_DEFAULT` is the documented pseudo-handle for the
    // current process symbol table. `c_name.as_ptr()` is a valid NUL-terminated
    // string for the duration of this call, and dlsym does not retain it.
    let ptr = unsafe { dl::dlsym(dl::RTLD_DEFAULT, c_name.as_ptr()) };
    if ptr.is_null() {
        None
    } else {
        Some(ptr as *const u8)
    }
}

#[cfg(not(unix))]
pub fn lookup_process_symbol(_name: &str) -> Option<*const u8> {
    None
}

fn resolve_extern(name: &str, extern_symbols: &HashMap<String, *const u8>) -> Option<*const u8> {
    if let Some(&ptr) = extern_symbols.get(name) {
        return Some(ptr);
    }
    if let Some(ptr) = lookup_process_symbol(name) {
        return Some(ptr);
    }
    #[cfg(target_os = "macos")]
    if let Some(stripped) = name.strip_prefix('_') {
        if let Some(ptr) = lookup_process_symbol(stripped) {
            return Some(ptr);
        }
    }
    None
}

pub struct JitCompiler {
    /// Pipeline constructed from the caller-supplied [`JitConfig`]. Held for
    /// forward compatibility: `compile_raw` does not currently invoke the
    /// pipeline, but the config (notably `verify_dispatch`, see #375) must
    /// be preserved so future heterogeneous-aware JIT entry points inherit
    /// the caller's policy. Also reached from unit tests via the
    /// `#[cfg(test)] pipeline()` accessor.
    #[allow(dead_code)]
    pipeline: Pipeline,
    profile_hooks: ProfileHookMode,
}

impl JitCompiler {
    pub fn new(config: JitConfig) -> Self {
        let profile_hooks =
            if config.profile_hooks == ProfileHookMode::None && config.emit_entry_counters {
                ProfileHookMode::CallCounts
            } else {
                config.profile_hooks
            };
        Self {
            pipeline: Pipeline::new(PipelineConfig {
                opt_level: config.opt_level,
                emit_debug: false,
                // #375: Propagate caller-visible dispatch-verification policy
                // instead of hard-coding FallbackOnFailure. The JitConfig
                // default is ErrorOnFailure so failures surface as errors.
                verify_dispatch: config.verify_dispatch,
                verify: config.verify,
                enable_post_ra_opt: config.opt_level != OptLevel::O0,
                use_pressure_aware_scheduler: matches!(
                    config.opt_level,
                    OptLevel::O2 | OptLevel::O3
                ),
                // CEGIS superopt is not scheduled by the JIT path; the
                // budget knob is only wired into the batch compiler (AOT).
                cegis_superopt_budget_sec: None,
                target_triple: String::new(),
            }),
            profile_hooks,
        }
    }

    /// Returns a reference to the underlying compilation pipeline.
    ///
    /// Test-only accessor (`#[cfg(test)]`) used by the dispatch-verification
    /// default regression tests (#375) to reach
    /// [`Pipeline::generate_and_verify_dispatch`] without adding a
    /// permanent public API surface. Not part of the supported JIT API.
    #[cfg(test)]
    pub(crate) fn pipeline(&self) -> &Pipeline {
        &self.pipeline
    }

    /// Compile post-regalloc IR functions to executable memory.
    ///
    /// `extern_symbols` maps mangled names (e.g., `"_helper"`) to host addresses.
    ///
    /// # Symbol uniqueness
    ///
    /// Every function's primary name and its `_`-prefixed Mach-O alias must
    /// be unique across the entire `functions` slice. A duplicate in either
    /// slot — e.g., two functions named `"foo"`, or one `"foo"` and one
    /// `"_foo"` (whose primary key collides with the first's alias) —
    /// returns [`JitError::DuplicateSymbol`] instead of silently
    /// overwriting the earlier function's offset (#374).
    pub fn compile_raw(
        &self,
        functions: &[IrMachFunction],
        extern_symbols: &HashMap<String, *const u8>,
    ) -> Result<ExecutableBuffer, JitError> {
        self.compile_raw_inner(functions, extern_symbols, |_func, _duration| {})
    }

    /// Internal variant of [`Self::compile_raw`] that also reports
    /// per-function encoding durations to the caller.
    pub(crate) fn compile_raw_with_encoding_metrics(
        &self,
        functions: &[IrMachFunction],
        extern_symbols: &HashMap<String, *const u8>,
    ) -> Result<(ExecutableBuffer, HashMap<String, Duration>), JitError> {
        let mut encoding_timings = HashMap::with_capacity(functions.len());
        let buffer = self.compile_raw_inner(functions, extern_symbols, |func, duration| {
            encoding_timings.insert(func.name.clone(), duration);
        })?;
        Ok((buffer, encoding_timings))
    }

    fn compile_raw_inner<F>(
        &self,
        functions: &[IrMachFunction],
        extern_symbols: &HashMap<String, *const u8>,
        mut record_encoding: F,
    ) -> Result<ExecutableBuffer, JitError>
    where
        F: FnMut(&IrMachFunction, Duration),
    {
        // #396 Phase 2: block/edge/frequency/loop-head modes are API
        // reserved but the trampoline emitter has not been extended
        // yet. Reject early so the caller gets a clear diagnostic
        // instead of silently producing unhooked code.
        if profile_hooks_is_phase2_stub(self.profile_hooks) {
            return Err(JitError::ProfileHookModeUnimplemented {
                mode: self.profile_hooks,
            });
        }

        let mut code = Vec::with_capacity(functions.len() * 128);
        let mut fixups: Vec<Fixup> = Vec::new();
        let mut func_offsets: HashMap<String, u64> = HashMap::new();
        let mut counters: HashMap<String, Box<AtomicU64>> = HashMap::new();
        // Per-block `{count, total_cycles}` cells, populated only under
        // `ProfileHookMode::BlockCountsAndTiming` (#364 Phase 3). Otherwise
        // left empty and handed off to the `ExecutableBuffer` as-is.
        let mut timing_cells: HashMap<String, Box<BlockTimingCell>> = HashMap::new();
        // Per-buffer `TimingState`. `Some` iff `BlockCountsAndTiming` is
        // enabled for this compile. Allocated upfront (before the first
        // trampoline is emitted) so the trampolines' literal-pool patches
        // can bake in a stable address.
        let mut timing_state: Option<Box<TimingState>> = None;
        // Canonical (user-provided) function names in insertion order. This is
        // the authoritative symbol list — `symbol_offsets` contains additional
        // alias keys (`"_foo"` for Mach-O dlsym compatibility), which are
        // lookup conveniences rather than independent symbols. (Fix #360.)
        let mut canonical_symbols: Vec<String> = Vec::with_capacity(functions.len());
        let profile_counters_enabled = profile_hooks_enable_counters(self.profile_hooks);
        let profile_block_counters_enabled =
            profile_hooks_enable_block_counters(self.profile_hooks);
        let profile_block_timing_enabled =
            profile_hooks_enable_block_timing(self.profile_hooks);

        if profile_counters_enabled && !cfg!(any(target_arch = "aarch64", target_arch = "x86_64")) {
            return Err(JitError::ProfileHooksUnsupported);
        }

        // #364 BlockCounts is implemented on AArch64 only in the initial
        // landing. x86-64 block-level splicing is tracked as a follow-up
        // because its branch encoding (rel32 Jcc / rel8 Jcc mix) needs a
        // separate re-patcher. Reject x86-64 BlockCounts with a clear
        // diagnostic rather than silently producing per-function trampolines
        // only.
        if profile_block_counters_enabled && !cfg!(target_arch = "aarch64") {
            return Err(JitError::ProfileHooksUnsupported);
        }

        // #364 Phase 3 BlockCountsAndTiming is likewise AArch64-only in the
        // initial landing — the timing trampoline uses the `MRS CNTVCT_EL0`
        // virtual-counter system register. An x86-64 port (using `RDTSC`)
        // is tracked as a follow-up.
        if profile_block_timing_enabled && !cfg!(target_arch = "aarch64") {
            return Err(JitError::ProfileHooksUnsupported);
        }

        // Allocate the single per-buffer `TimingState` upfront so the
        // trampolines' literal slots can capture a stable address.
        if profile_block_timing_enabled {
            timing_state = Some(Box::new(TimingState {
                prev_ts: AtomicU64::new(0),
                prev_accum_ptr: AtomicU64::new(0),
            }));
        }

        let mut counter_patch_sites: Vec<(usize, *const AtomicU64)> =
            Vec::with_capacity(functions.len());
        // Per-buffer `&TimingState` patch sites. Populated only under
        // `BlockCountsAndTiming`; drained after the code buffer is mapped.
        let mut tstate_patch_sites: Vec<usize> = Vec::new();

        // Per-function byte ranges in the code buffer. Used by the proof
        // certificate path (issue #348) to tell callers which bytes each
        // certified function occupies. Populated regardless of whether
        // verification is enabled; certificate construction itself is
        // gated on `self.pipeline` config below.
        let mut func_ranges: Vec<(String, std::ops::Range<u64>)> =
            Vec::with_capacity(functions.len());

        // Encode all functions into a contiguous buffer.
        for func in functions {
            let start = code.len() as u64;
            let mut body_start = start;
            canonical_symbols.push(func.name.clone());

            if func_offsets.contains_key(func.name.as_str()) {
                return Err(JitError::DuplicateSymbol(func.name.clone()));
            }
            func_offsets.insert(func.name.clone(), start);

            // Insert the underscore-prefixed alias so Mach-O-style lookups
            // (`"_foo"`) resolve to the same offset. If the canonical name
            // already begins with `_` this happens to produce `"__foo"`,
            // which is harmless: it is never consulted by `symbols()` or
            // `symbol_count()` because those iterate `canonical_symbols`.
            //
            // The "Mach-O-style" naming here refers to a caller convention
            // (the leading underscore C symbols get under darwin), not to an
            // object-file format. On Linux hosts (#346) this alias is a
            // dormant extra key in the lookup map unless a caller explicitly
            // asks for the `_`-prefixed form, so leaving it in place keeps
            // the JIT's public symbol API identical across macOS and Linux.
            let alias = format!("_{}", func.name);
            if func_offsets.contains_key(alias.as_str()) {
                return Err(JitError::DuplicateSymbol(alias));
            }
            func_offsets.insert(alias, start);

            if profile_counters_enabled {
                let counter = Box::new(AtomicU64::new(0));
                let counter_ptr = counter.as_ref() as *const AtomicU64;
                counters.insert(func.name.clone(), counter);
                if cfg!(target_arch = "aarch64") {
                    #[cfg(target_arch = "aarch64")]
                    {
                        let literal_slot_offset = emit_profile_trampoline_aarch64(&mut code);
                        counter_patch_sites.push((literal_slot_offset, counter_ptr));
                    }
                } else if cfg!(target_arch = "x86_64") {
                    #[cfg(target_arch = "x86_64")]
                    {
                        let imm64_offset = emit_profile_trampoline_x86_64(&mut code);
                        counter_patch_sites.push((imm64_offset, counter_ptr));
                    }
                } else {
                    return Err(JitError::ProfileHooksUnsupported);
                }
                body_start = code.len() as u64;
            }

            let encode_start = Instant::now();
            let (bytes, fxs) = if profile_block_counters_enabled {
                // #364 BlockCounts path (AArch64-only at present).
                //
                // 1. Encode the function normally and capture per-block byte
                //    offsets so we can splice in a trampoline at the start
                //    of every block.
                // 2. Allocate one `AtomicU64` per basic block, keyed as
                //    `"{func.name}::block{block_id.0}"`. The entry block's
                //    counter doubles as the function-entry counter and is
                //    re-exposed under `func.name` via a read-side alias on
                //    `ExecutableBuffer::get_profile` / `entry_count`.
                // 3. Run `splice_block_trampolines_aarch64`, which returns
                //    the spliced bytes plus the list of
                //    `(block_id, literal_slot_offset_within_spliced_bytes)`
                //    patch sites. Fixup offsets are shifted in-place so
                //    external symbol fixups still index the correct branch
                //    instruction.
                // 4. Register each patch site against `counter_patch_sites`
                //    so the late-binding code that writes counter pointers
                //    into the mmap'd buffer picks them up uniformly with
                //    the per-function trampolines.
                #[cfg(target_arch = "aarch64")]
                {
                    let (body_bytes, mut block_fixups, block_byte_offsets) =
                        encode_function_with_fixups_and_blocks(func)?;
                    record_encoding(func, encode_start.elapsed());

                    // Allocate per-block counters.
                    let mut block_counter_ptrs: HashMap<BlockId, *const AtomicU64> =
                        HashMap::with_capacity(func.block_order.len());
                    for &bid in func.block_order.iter() {
                        let key = format!("{}::block{}", func.name, bid.0);
                        if counters.contains_key(&key) {
                            return Err(JitError::DuplicateSymbol(key));
                        }
                        let counter = Box::new(AtomicU64::new(0));
                        let counter_ptr = counter.as_ref() as *const AtomicU64;
                        block_counter_ptrs.insert(bid, counter_ptr);
                        counters.insert(key, counter);
                    }

                    let (spliced, tramp_sites) = splice_block_trampolines_aarch64(
                        func,
                        &body_bytes,
                        &block_byte_offsets,
                        block_fixups.as_mut_slice(),
                    )?;

                    // Register patch sites: translate each
                    // (block_id, literal_slot_offset_within_spliced) into
                    // (buffer_absolute_offset, counter_ptr).
                    let func_base = code.len();
                    for (bid, slot_off) in tramp_sites {
                        let ptr = block_counter_ptrs
                            .get(&bid)
                            .copied()
                            .expect("every block has an allocated counter");
                        counter_patch_sites.push((func_base + slot_off, ptr));
                    }

                    // The first block's trampoline IS the function entry
                    // point, so body_start must point at the start of the
                    // spliced region (not past it). The per-function
                    // trampoline emitted above when `profile_counters_enabled`
                    // is mutually exclusive with BlockCounts, so body_start
                    // is still `start` here.
                    debug_assert!(!profile_counters_enabled);
                    (spliced, block_fixups)
                }
                #[cfg(not(target_arch = "aarch64"))]
                {
                    // Unreachable: guarded by the top-of-function arch
                    // check. Included so this branch type-checks on all
                    // architectures.
                    unreachable!("profile_block_counters_enabled implies target_arch = aarch64")
                }
            } else if profile_block_timing_enabled {
                // #364 Phase 3 BlockCountsAndTiming path (AArch64-only).
                //
                // Mirrors the plain-counter path above but:
                // - Allocates one `BlockTimingCell {count, total_cycles}`
                //   per basic block instead of a single counter.
                // - Calls `splice_block_trampolines_with_timing_aarch64`,
                //   which emits 100-byte timing trampolines and returns
                //   two patch-site offsets per block (counter literal and
                //   `TimingState` literal).
                // - Pushes each (counter_ptr) site onto
                //   `counter_patch_sites` and each (tstate) site onto
                //   `tstate_patch_sites`; both lists are drained after
                //   the mmap is written, identical in shape to the plain
                //   counter path.
                #[cfg(target_arch = "aarch64")]
                {
                    let (body_bytes, mut block_fixups, block_byte_offsets) =
                        encode_function_with_fixups_and_blocks(func)?;
                    record_encoding(func, encode_start.elapsed());

                    // Allocate per-block timing cells. Capture raw pointers
                    // to the `count` field — the trampoline increments
                    // `count` at offset 0 of the cell, so the LDR/ADD/STR
                    // targets the cell's start address directly.
                    let mut block_cell_ptrs: HashMap<BlockId, *const AtomicU64> =
                        HashMap::with_capacity(func.block_order.len());
                    for &bid in func.block_order.iter() {
                        let key = format!("{}::block{}", func.name, bid.0);
                        if counters.contains_key(&key) || timing_cells.contains_key(&key) {
                            return Err(JitError::DuplicateSymbol(key));
                        }
                        let cell = Box::new(BlockTimingCell {
                            count: AtomicU64::new(0),
                            total_cycles: AtomicU64::new(0),
                        });
                        // `&cell.count` sits at offset 0 of a `#[repr(C)]`
                        // BlockTimingCell, so its address equals the cell's
                        // address. The trampoline writes `total_cycles` at
                        // cell + 8 via an explicit `ADD X11, X16, #8`.
                        let cell_ptr = &cell.count as *const AtomicU64;
                        block_cell_ptrs.insert(bid, cell_ptr);
                        timing_cells.insert(key, cell);
                    }

                    let (spliced, tramp_sites) =
                        splice_block_trampolines_with_timing_aarch64(
                            func,
                            &body_bytes,
                            &block_byte_offsets,
                            block_fixups.as_mut_slice(),
                        )?;

                    // Register patch sites for both the counter-cell literal
                    // and the timing-state literal of each block.
                    let func_base = code.len();
                    for (bid, counter_off, tstate_off) in tramp_sites {
                        let ptr = block_cell_ptrs
                            .get(&bid)
                            .copied()
                            .expect("every block has an allocated timing cell");
                        counter_patch_sites.push((func_base + counter_off, ptr));
                        tstate_patch_sites.push(func_base + tstate_off);
                    }

                    debug_assert!(!profile_counters_enabled);
                    (spliced, block_fixups)
                }
                #[cfg(not(target_arch = "aarch64"))]
                {
                    unreachable!("profile_block_timing_enabled implies target_arch = aarch64")
                }
            } else {
                let (bytes, fxs) = encode_function_with_fixups(func)?;
                record_encoding(func, encode_start.elapsed());
                (bytes, fxs)
            };
            for fx in fxs.iter() {
                let mut adjusted = fx.clone();
                adjusted.offset += body_start as u32;
                fixups.push(adjusted);
            }
            code.extend_from_slice(&bytes);
            let end = code.len() as u64;
            // When profile hooks are enabled the callable entry point starts
            // at the trampoline, so the certified range must cover the full
            // compiled region from trampoline start through body end.
            func_ranges.push((func.name.clone(), start..end));
        }

        // Resolve internal fixups.
        for fixup in &fixups {
            let addr = match &fixup.target {
                FixupTarget::NamedSymbol(name) => {
                    if let Some(&off) = func_offsets.get(name) {
                        off
                    } else if resolve_extern(name, extern_symbols).is_some() {
                        continue;
                    } else {
                        return Err(JitError::UnresolvedSymbol(name.clone()));
                    }
                }
                FixupTarget::Symbol(idx) => {
                    let name = functions
                        .get(*idx as usize)
                        .map(|f| &f.name)
                        .ok_or_else(|| JitError::UnresolvedSymbol(format!("index {}", idx)))?;
                    *func_offsets.get(name.as_str()).unwrap()
                }
                _ => continue,
            };
            patch_fixup(&mut code, fixup.offset, addr)?;
        }

        // Build veneer trampolines for external calls and patch their BL sites.
        //
        // Fix #362: single-pass over fixups — no intermediate clone+Vec+double
        // iteration. `HashMap::entry` dedups veneers in O(1) per fixup while
        // preserving deterministic, first-seen emission order (driven by the
        // deterministic order of `fixups`).
        //
        // Fix #345: on AArch64, BL has +-128 MiB reach. Veneers are appended
        // after all function code; if the module is large, a BL at the start
        // of the buffer may not be able to reach a veneer at the end. We
        // pre-validate each external BL's distance to its veneer and return a
        // typed `VeneerOutOfRange` error instead of letting `patch_branch26`
        // emit a corrupt instruction or surface the generic `BranchOutOfRange`.
        let mut veneers: HashMap<String, u64> = HashMap::new();
        // Deferred patches: (fixup_offset, veneer_offset, symbol_name). We
        // cannot patch while still emitting veneers because emitting grows
        // `code`, and each veneer's final offset is only known at emission
        // time. Collecting the (site, veneer) pairs in one pass lets us patch
        // after all veneers are laid out.
        let mut ext_patches: Vec<(u32, u64, String)> = Vec::new();
        for fixup in &fixups {
            let name = match &fixup.target {
                FixupTarget::NamedSymbol(n) if !func_offsets.contains_key(n) => n,
                _ => continue,
            };
            let veneer_off = *veneers.entry(name.clone()).or_insert_with(|| {
                let pos = code.len() as u64;
                emit_veneer_stub(&mut code);
                pos
            });
            ext_patches.push((fixup.offset, veneer_off, name.clone()));
        }

        // Pre-validate BL reachability for every external fixup before we
        // start mutating the instruction stream. Patching is all-or-nothing:
        // if any BL is out of range we bail out with a descriptive error
        // rather than leaving the buffer half-patched. The check is factored
        // out into `validate_veneer_ranges` so #345 can be regression-tested
        // without emitting >128 MiB of real code.
        validate_veneer_ranges(&ext_patches, code.len())?;

        for (fx_off, veneer_off, _sym) in &ext_patches {
            patch_fixup(&mut code, *fx_off, *veneer_off)?;
        }

        // Allocate RW memory, copy code, patch external addresses, then RW→RX.
        let alloc_size = sys::page_align(code.len());
        let memory = unsafe { sys::mmap(alloc_size, sys::RW).map_err(JitError::MemoryAlloc)? };

        unsafe {
            // SAFETY: `memory` points to a writable allocation returned by mmap,
            // `code.as_ptr()` is valid for `code.len()` bytes, and the regions do
            // not overlap because `code` is heap memory distinct from the mmap.
            std::ptr::copy_nonoverlapping(code.as_ptr(), memory, code.len());
        }

        // Lifetime-invariant check (#494): each baked-in counter pointer
        // MUST point into a `Box` that this function is about to transfer
        // ownership of to the returned `ExecutableBuffer`. Otherwise the
        // trampoline would dereference freed memory the moment this
        // function returns. Build a debug-only set of valid Box addresses
        // (`counters` + `timing_cells.count` fields) and verify every
        // patch site lands in it.
        #[cfg(debug_assertions)]
        {
            use std::collections::HashSet;
            let mut valid: HashSet<usize> = HashSet::with_capacity(
                counters.len() + timing_cells.len(),
            );
            for counter in counters.values() {
                valid.insert(counter.as_ref() as *const AtomicU64 as usize);
            }
            for cell in timing_cells.values() {
                // The `count` field is the one referenced by per-block
                // trampolines under `BlockCounts` and the counter slot of
                // `BlockCountsAndTiming`. `total_cycles` is addressed via
                // `TimingState::prev_accum_ptr` (stored at runtime, not at
                // compile time), so it is NOT a patch-site target.
                valid.insert(&cell.count as *const AtomicU64 as usize);
            }
            for (patch_offset, counter_ptr) in &counter_patch_sites {
                debug_assert!(
                    !counter_ptr.is_null(),
                    "counter patch site at offset {} has null counter pointer",
                    patch_offset
                );
                debug_assert!(
                    valid.contains(&(*counter_ptr as usize)),
                    "counter patch site at offset {} points outside the \
                     ExecutableBuffer-owned counter/timing_cell boxes \
                     (counter_ptr = {:p}); would dangle on return (see #494)",
                    patch_offset,
                    *counter_ptr
                );
            }
        }

        for (patch_offset, counter_ptr) in &counter_patch_sites {
            unsafe {
                // SAFETY: `patch_offset` points at the 8-byte immediate /
                // literal slot inside a just-copied trampoline, and
                // `counter_ptr` points at the owned `AtomicU64` backing
                // that trampoline. The debug_assert above verifies that
                // `counter_ptr` lands in a `Box` being transferred into
                // the returned `ExecutableBuffer` (#494 invariant).
                std::ptr::write(memory.add(*patch_offset) as *mut u64, *counter_ptr as u64);
            }
        }

        // Patch the `&TimingState` literal slot in every
        // `BlockCountsAndTiming` trampoline. The `timing_state` allocation
        // is owned by the `ExecutableBuffer` so the baked-in pointer stays
        // valid for the lifetime of the executable mapping.
        if let Some(tstate) = timing_state.as_ref() {
            let tstate_ptr = &**tstate as *const TimingState as u64;
            for patch_offset in &tstate_patch_sites {
                unsafe {
                    // SAFETY: `patch_offset` points at the 8-byte literal
                    // slot for the timing-state pointer inside a
                    // just-copied timing trampoline. `tstate_ptr` refers to
                    // the owned `Box<TimingState>` below.
                    std::ptr::write(memory.add(*patch_offset) as *mut u64, tstate_ptr);
                }
            }
        } else {
            debug_assert!(
                tstate_patch_sites.is_empty(),
                "tstate patch sites registered without an allocated TimingState"
            );
        }

        for (name, &veneer_off) in &veneers {
            let ext_addr = resolve_extern(name, extern_symbols).ok_or_else(|| {
                // SAFETY: `memory`/`alloc_size` refer to the live mmap allocation
                // created above and have not been unmapped yet on this path.
                unsafe {
                    sys::munmap(memory, alloc_size);
                }
                JitError::UnresolvedSymbol(name.clone())
            })?;
            unsafe {
                let slot = memory.add(veneer_off as usize + veneer_addr_offset()) as *mut u64;
                // SAFETY: `veneer_off` was produced from `code.len()` before the
                // copy into `memory`, and `veneer_addr_offset()` lands on the
                // embedded 64-bit address slot within that veneer stub.
                *slot = ext_addr as u64;
            }
        }

        unsafe {
            // Flush data+instruction cache while page is still RW. The ARM ARM
            // sequence is:
            //   1. dc cvau per line (clean D-cache to Point-of-Unification)
            //   2. dsb ish
            //   3. ic ivau per line (invalidate I-cache)
            //   4. dsb ish
            //   5. isb
            // All 5 steps are inside `sys::flush_icache`.
            //
            // Running this before `mprotect` RW->RX is the architecturally-required
            // ordering. On Apple Silicon with MAP_JIT the hardware tolerates flushing
            // after mprotect, but stricter implementations can fault because
            // `dc cvau` on an executable-only page without read permission is UB.
            // See: ARM Architecture Reference Manual B2.4.4 "Cache maintenance
            // instructions", and Apple's JIT-compilation guide for MAP_JIT pages.
            // Issue #357.
            //
            // SAFETY: `memory`/`alloc_size` refer to a valid mmap allocation that
            // is transitioning from RW to RX after all writes are complete.
            sys::flush_icache(memory, code.len());
            sys::mprotect(memory, alloc_size, sys::RX).map_err(|e| {
                // SAFETY: If mprotect fails, the mmap allocation is still live and
                // must be released before returning the error.
                sys::munmap(memory, alloc_size);
                JitError::MemoryProtect(e)
            })?;
        }

        // Proof-certificate path (issue #348).
        //
        // When `JitConfig::verify` is set we run llvm2-verify's
        // `verify_function` for each compiled function and attach a
        // `JitCertificate` (containing the full `CertificateChain` plus
        // coarse tMIR provenance) to the `ExecutableBuffer`. Callers such
        // as tla2 can then ask `buffer.certificate(name)` to prove that
        // the JIT'd machine code was verified against its tMIR input. The
        // recorded byte range already includes any prepended profile
        // trampoline so the certificate covers the full callable region.
        //
        // When `verify` is off, `certificates` stays empty and the
        // `ExecutableBuffer::certificate` accessor returns `None`, keeping
        // the fast-path cost of JIT compilation unchanged.
        #[cfg(feature = "verify")]
        let certificates: HashMap<String, crate::jit_cert::JitCertificate> =
            if self.pipeline.config.verify {
                let mut certs = HashMap::with_capacity(functions.len());
                for (func, (_name, range)) in functions.iter().zip(func_ranges.iter()) {
                    let report = llvm2_verify::verify_function(func);
                    let cert =
                        crate::jit_cert::JitCertificate::from_report(func, &report, range.clone());
                    certs.insert(func.name.clone(), cert);
                }
                certs
            } else {
                HashMap::new()
            };

        #[cfg(not(feature = "verify"))]
        let certificates: HashMap<String, crate::jit_cert::JitCertificate> = HashMap::new();

        Ok(ExecutableBuffer {
            memory,
            len: alloc_size,
            symbol_offsets: func_offsets,
            canonical_symbols,
            counters,
            timing_cells,
            timing_state,
            certificates,
        })
    }
}

/// A typed JIT function pointer tied to the lifetime of the owning
/// [`ExecutableBuffer`]. Cannot outlive the buffer by construction.
///
/// ```compile_fail
/// use llvm2_codegen::jit::{JitCompiler, JitConfig};
/// use std::collections::HashMap;
///
/// let jit = JitCompiler::new(JitConfig::default());
/// let ext: HashMap<String, *const u8> = HashMap::new();
/// let buf = jit.compile_raw(&[], &ext).unwrap();
/// let func = unsafe { buf.get_fn_bound::<extern "C" fn()>("foo") };
/// drop(buf);
/// let _ = func;
/// ```
#[derive(Copy, Clone)]
pub struct JitFn<'a, F: Copy> {
    inner: F,
    _marker: PhantomData<&'a ExecutableBuffer>,
}

impl<'a, F: Copy> JitFn<'a, F> {
    /// Returns the underlying `F`. Still safe because `F` itself is
    /// typically an `extern "C" fn(...)` — which is `Copy + 'static` —
    /// so the lifetime exists only on the wrapper. Callers that leak
    /// the inner `F` past the buffer's lifetime re-enter the unsafe
    /// world. Prefer `as_ref` / keep the `JitFn` wrapper in scope.
    pub fn into_inner(self) -> F {
        self.inner
    }

    /// Returns a reference to the underlying `F` without consuming the
    /// lifetime guard. This is the recommended way to use a `JitFn`.
    pub fn as_ref(&self) -> &F {
        &self.inner
    }
}

/// A raw code pointer tied to the lifetime of the owning buffer.
///
/// ```compile_fail
/// use llvm2_codegen::jit::{JitCompiler, JitConfig};
/// use std::collections::HashMap;
///
/// let jit = JitCompiler::new(JitConfig::default());
/// let ext: HashMap<String, *const u8> = HashMap::new();
/// let buf = jit.compile_raw(&[], &ext).unwrap();
/// let ptr = buf.get_fn_ptr_bound("foo");
/// drop(buf);
/// let _ = ptr;
/// ```
#[derive(Copy, Clone)]
pub struct JitPtr<'a> {
    ptr: *const u8,
    _marker: PhantomData<&'a ExecutableBuffer>,
}

impl<'a> JitPtr<'a> {
    pub fn as_ptr(self) -> *const u8 {
        self.ptr
    }
}

/// Per-basic-block `{count, total_cycles}` cell used by
/// [`ProfileHookMode::BlockCountsAndTiming`] (#364 Phase 3).
///
/// The AArch64 timing trampoline emits LDR/ADD/STR against the `count`
/// field at offset 0 and against the `total_cycles` field at offset 8, so
/// the layout is `#[repr(C)]`-pinned and must NOT be reordered. Both
/// fields are accessed with relaxed atomics from the trampoline, matching
/// the Rust-side `Ordering::Relaxed` used by the reader accessors.
///
/// # Lifetime invariant (#494)
///
/// Each `BlockTimingCell` is allocated as `Box<BlockTimingCell>` and
/// stored in [`ExecutableBuffer::timing_cells`]. The AArch64 timing
/// trampoline bakes the cell's `Box`-pinned heap address into its
/// literal pool, so the cell **must outlive the executable mapping**
/// it is referenced from. This is guaranteed structurally: `Drop` for
/// [`ExecutableBuffer`] unmaps `memory` before Rust drops the owning
/// `HashMap` (field declaration order). See the module-level
/// "Profile counter & timing-cell lifetime" section for the full
/// contract. Related: #478 (per-function counter path),
/// #364 (block-level extension), #494 (this lifetime documentation).
#[repr(C)]
pub(crate) struct BlockTimingCell {
    /// Block entry count. Incremented on every entry by the trampoline.
    /// Equivalent to the single counter used by
    /// [`ProfileHookMode::BlockCounts`].
    pub(crate) count: AtomicU64,
    /// Accumulated cycles spent in this block, measured as the sum of
    /// `(next_block_entry_ts - this_block_entry_ts)` deltas attributed
    /// back to this cell by subsequent block-entry trampolines. See
    /// [`TimingState`] for the attribution machinery.
    pub(crate) total_cycles: AtomicU64,
}

/// Shared cross-block attribution state for
/// [`ProfileHookMode::BlockCountsAndTiming`] (#364 Phase 3).
///
/// One instance per [`ExecutableBuffer`]. On each block entry the
/// trampoline:
/// 1. Reads `prev_ts`. If zero (first block ever entered under this
///    buffer), the attribution step is skipped.
/// 2. Otherwise computes `delta = now - prev_ts` and accumulates it into
///    the [`BlockTimingCell::total_cycles`] field pointed to by
///    `prev_accum_ptr`.
/// 3. Writes `prev_ts = now` and `prev_accum_ptr = &cell.total_cycles`
///    for the current block.
///
/// The layout is `#[repr(C)]` because the trampoline accesses `prev_ts`
/// at offset 0 and `prev_accum_ptr` at offset 8 via fixed LDR/STR
/// immediates.
///
/// **Concurrency limitation (documented, intentional for Phase 3):** the
/// state is a single buffer-wide pair, accessed with relaxed atomics. On
/// a single thread this gives the intended per-block total cycle
/// attribution. With multiple threads calling into the same buffer
/// concurrently, the attribution races — a thread's block may be charged
/// for another thread's cycles. The `count` field is still correct; only
/// `total_cycles` is racy. A per-thread `TimingState` is a straightforward
/// follow-up (pthread_self keying or TLS slot), filed as a gap in the
/// issue comment for #364 Phase 3.
///
/// # Lifetime invariant (#494)
///
/// The `TimingState` is allocated as `Box<TimingState>` once per buffer
/// and stored in [`ExecutableBuffer::timing_state`]. Every timing
/// trampoline in the buffer bakes its heap address into its literal
/// pool, so the `TimingState` **must outlive the executable mapping**.
/// The guarantee is structural, not convention: `Drop` for
/// [`ExecutableBuffer`] unmaps `memory` before Rust drops
/// `timing_state`, so the trampolines cannot execute against a freed
/// allocation. See the module-level "Profile counter & timing-cell
/// lifetime" section for the full contract. Related: #364, #494.
#[repr(C)]
pub(crate) struct TimingState {
    /// Timestamp (`CNTVCT_EL0`) captured at the last block entry, or 0
    /// if no block has been entered yet under this buffer.
    pub(crate) prev_ts: AtomicU64,
    /// Raw pointer to the `total_cycles` field of the
    /// [`BlockTimingCell`] for the previously-entered block, or 0 if
    /// none. Stored as `u64` so the trampoline can load it directly with
    /// an `LDR`. Cast back to `*mut AtomicU64` only inside the buffer
    /// while the mapping is alive.
    pub(crate) prev_accum_ptr: AtomicU64,
}

/// Executable memory buffer containing compiled native functions.
///
/// The generated code remains valid only while this buffer is alive. Prefer
/// [`Self::get_fn_bound`] and [`Self::get_fn_ptr_bound`], which tie returned
/// handles to `&self` so they cannot outlive the mapping. The legacy
/// [`Self::get_fn`] and [`Self::get_fn_ptr`] APIs return raw values with no
/// lifetime tracking, so callers must keep the buffer alive for the full
/// duration of any outstanding function pointer or code pointer.
///
/// # Field drop order (profile counter lifetime — #494)
///
/// `memory` is declared first, so it is unmapped (via `Drop for
/// ExecutableBuffer`, which calls `munmap`) strictly before any of the
/// heap-allocated counter / timing-cell / timing-state fields are
/// dropped. The AArch64 profile trampolines bake the counters'
/// `Box`-pinned addresses into the code buffer's literal pool, so the
/// trampolines cannot execute against a freed counter after `munmap`:
/// the text page containing the literal is gone first. Reordering these
/// fields WOULD INTRODUCE A USE-AFTER-FREE WINDOW during buffer
/// teardown — do not reorder.
///
/// See the module-level "Profile counter & timing-cell lifetime"
/// section for the full contract. Related: #478, #364, #494.
pub struct ExecutableBuffer {
    // NOTE (#494): `memory` MUST remain the first field so `Drop` unmaps
    // it before the counter / timing boxes are dropped. See the struct-
    // level doc comment above for the full ordering argument.
    memory: *mut u8,
    len: usize,
    /// Symbol → offset lookup table. Contains both canonical function names
    /// (as supplied by the caller) and Mach-O-style `_name` aliases pointing
    /// to the same offset. This map is a lookup convenience for
    /// `get_fn_ptr` / `get_fn_bound`; it is NOT the canonical symbol list.
    /// Use `canonical_symbols` (via `symbols()` / `symbol_count()`) for the
    /// authoritative, de-duplicated symbol view (fix #360).
    symbol_offsets: HashMap<String, u64>,
    /// Canonical, user-provided function names in insertion order. One entry
    /// per compiled function. Distinct from `symbol_offsets` which also holds
    /// underscore-prefixed aliases.
    canonical_symbols: Vec<String>,
    /// Per-function profiling counters keyed by canonical name.
    ///
    /// Under [`ProfileHookMode::BlockCounts`] / `BlockCountsAndTiming` this
    /// map ALSO carries per-block counters keyed as
    /// `"{func_name}::block{block_id.0}"` (issue #364). Reader accessors
    /// [`Self::get_profile`] / [`Self::entry_count`] fall back to the
    /// entry-block alias when the per-function key is absent.
    ///
    /// # Lifetime invariant (#478, #494)
    ///
    /// The code buffer bakes raw `*const AtomicU64` pointers — the
    /// `Box`-pinned heap addresses returned by
    /// `Box::as_ref() as *const AtomicU64` — into any emitted entry /
    /// block trampolines. The buffer owns the `Box` allocations for the
    /// full lifetime of the executable mapping: `Drop` for
    /// [`ExecutableBuffer`] unmaps `memory` before Rust drops this map
    /// (declared after `memory`), so the trampolines cannot execute
    /// against a freed counter. See the module-level "Profile counter
    /// & timing-cell lifetime" section for the full contract.
    counters: HashMap<String, Box<AtomicU64>>,
    /// Per-basic-block `{count, total_cycles}` cells keyed as
    /// `"{func_name}::block{block_id.0}"`. Populated only when the buffer
    /// was compiled with [`ProfileHookMode::BlockCountsAndTiming`]
    /// (#364 Phase 3). The AArch64 timing trampoline bakes a raw pointer
    /// to each cell into the code buffer.
    ///
    /// # Lifetime invariant (#494)
    ///
    /// These allocations MUST outlive the executable mapping — which they
    /// do, since the `ExecutableBuffer` owns both and `memory` is dropped
    /// first by `Drop for ExecutableBuffer` (field declaration order). See
    /// the module-level "Profile counter & timing-cell lifetime" section
    /// for the full contract.
    timing_cells: HashMap<String, Box<BlockTimingCell>>,
    /// Buffer-wide cross-block timing attribution state. `Some` iff the
    /// buffer was compiled with [`ProfileHookMode::BlockCountsAndTiming`].
    /// The single allocation is addressed by every timing trampoline in
    /// the buffer, so lifetime-tying it to the buffer is mandatory — see
    /// [`TimingState`] for the invariant it maintains.
    ///
    /// # Lifetime invariant (#494)
    ///
    /// The trampolines bake `&*timing_state` as a raw pointer into their
    /// literal pools. The allocation MUST outlive the executable mapping.
    /// This is guaranteed structurally: `Drop for ExecutableBuffer`
    /// unmaps `memory` before Rust drops this field (declared after
    /// `memory`). See the module-level "Profile counter & timing-cell
    /// lifetime" section for the full contract.
    ///
    /// `dead_code` is allowed because the field's purpose is purely
    /// ownership: the live reference to this allocation is the raw `u64`
    /// pointer baked into every timing trampoline's literal pool at
    /// `compile_raw_inner` time. Rust cannot see that use, and adding a
    /// token reader would misrepresent the invariant (`TimingState` is
    /// mutated only by emitted machine code, never by Rust code).
    #[allow(dead_code)]
    timing_state: Option<Box<TimingState>>,
    /// Per-function proof certificates (issue #348).
    ///
    /// Populated when `JitConfig::verify` is true at compile time. Callers
    /// such as tla2 can query a specific function via
    /// [`ExecutableBuffer::certificate`], iterate all attached certificates
    /// via [`ExecutableBuffer::certificates`], or check the verify-all-
    /// functions invariant via [`ExecutableBuffer::all_verified`].
    ///
    /// When verification is disabled (runtime flag cleared or `verify`
    /// feature off), this map stays empty.
    certificates: HashMap<String, crate::jit_cert::JitCertificate>,
}

/*
SAFETY: After construction the mapped code buffer is immutable executable
memory, so transferring or sharing `ExecutableBuffer` itself across threads
does not create data races.

Issue #355 exposed a separate lifetime hazard: the legacy `get_fn_ptr` and
`get_fn` APIs return raw function pointers with no lifetime tie to `&self`.
That means safe-ish code can move those pointers to other threads and drop the
buffer while a call is still outstanding, causing use-after-free when `Drop`
unmaps the executable pages.

The lifetime-bound `get_fn_ptr_bound` / `get_fn_bound` APIs close that gap for
new code by making outstanding pointers borrow the buffer. Callers who use the
legacy raw APIs across threads are responsible for synchronizing buffer
lifetime with all outstanding pointers, for example by keeping the buffer alive
in an `Arc<ExecutableBuffer>` for the full duration of any call and only
dropping it after all threads have returned.
*/
unsafe impl Send for ExecutableBuffer {}
unsafe impl Sync for ExecutableBuffer {}

impl ExecutableBuffer {
    /// Lifetime-bound version of [`Self::get_fn_ptr`]. The returned
    /// [`JitPtr`] cannot outlive `self`, eliminating the use-after-free
    /// risk when the buffer is dropped while a pointer is held.
    pub fn get_fn_ptr_bound<'a>(&'a self, name: &str) -> Option<JitPtr<'a>> {
        self.symbol_offsets.get(name).map(|&off| JitPtr {
            ptr: unsafe { self.memory.add(off as usize) as *const u8 },
            _marker: PhantomData,
        })
    }

    #[deprecated = "use get_fn_ptr_bound to tie the code pointer to the ExecutableBuffer lifetime"]
    pub fn get_fn_ptr(&self, name: &str) -> Option<*const u8> {
        self.get_fn_ptr_bound(name).map(JitPtr::as_ptr)
    }

    /// Lifetime-bound version of [`Self::get_fn`]. The returned
    /// [`JitFn`] cannot outlive `self`.
    ///
    /// # Safety
    /// `F` must match the compiled function's ABI and be pointer-sized.
    pub unsafe fn get_fn_bound<'a, F: Copy>(&'a self, name: &str) -> Option<JitFn<'a, F>> {
        assert_eq!(
            std::mem::size_of::<F>(),
            std::mem::size_of::<*const u8>(),
            "get_fn_bound<F>: F must be pointer-sized (expected {} bytes, got {} bytes)",
            std::mem::size_of::<*const u8>(),
            std::mem::size_of::<F>(),
        );
        self.get_fn_ptr_bound(name).map(|ptr| {
            let raw = ptr.as_ptr();
            // SAFETY: caller asserts `F` is ABI-compatible with the compiled
            // function pointer (documented on `unsafe fn get_fn_bound`), and
            // the above size assertion pins `F` to pointer width.
            JitFn {
                inner: unsafe { std::mem::transmute_copy(&raw) },
                _marker: PhantomData,
            }
        })
    }

    #[deprecated = "use get_fn_bound to tie the function pointer to the ExecutableBuffer lifetime"]
    /// # Safety
    /// Caller must ensure `F` matches the compiled function's ABI.
    ///
    /// # Panics
    /// Panics if `size_of::<F>() != size_of::<*const u8>()` (F must be pointer-sized).
    pub unsafe fn get_fn<F>(&self, name: &str) -> Option<F> {
        assert_eq!(
            std::mem::size_of::<F>(),
            std::mem::size_of::<*const u8>(),
            "get_fn<F>: F must be pointer-sized (expected {} bytes, got {} bytes)",
            std::mem::size_of::<*const u8>(),
            std::mem::size_of::<F>(),
        );
        self.symbol_offsets.get(name).map(|&off| {
            // SAFETY: `off` is within `self.memory`'s allocation (produced by
            // the JIT writer); pointer-size assert above pins `F`'s layout.
            unsafe {
                let ptr = self.memory.add(off as usize) as *const u8;
                std::mem::transmute_copy(&ptr)
            }
        })
    }

    pub fn allocated_size(&self) -> usize {
        self.len
    }

    /// Number of distinct functions compiled into this buffer.
    ///
    /// Fix #360: previously computed as `symbol_offsets.len() / 2`, which
    /// relied on the fragile invariant that every canonical name had exactly
    /// one `_`-prefixed alias. That assumption broke for user names already
    /// starting with `_` and for any external mutation of `symbol_offsets`.
    /// Now returns the exact length of the canonical symbol list, which is
    /// populated once per compiled function.
    pub fn symbol_count(&self) -> usize {
        self.canonical_symbols.len()
    }

    /// Iterate canonical function names paired with their offset in the code
    /// buffer. Yields each compiled function exactly once, using the
    /// user-provided name (never the Mach-O `_`-prefixed alias).
    ///
    /// Fix #360: previous implementation filtered `symbol_offsets` by
    /// `!starts_with('_')`, which silently hid functions whose canonical
    /// name already began with `_`. Iterating the canonical list is both
    /// correct and O(n) without a hash probe per item.
    pub fn symbols(&self) -> impl Iterator<Item = (&str, u64)> {
        self.canonical_symbols.iter().map(move |name| {
            // Safety: every canonical name is inserted into `symbol_offsets`
            // at construction time. If this unwrap ever fires we have a
            // construction-time bug, not a user-input bug.
            let off = *self
                .symbol_offsets
                .get(name.as_str())
                .expect("canonical symbol missing from symbol_offsets");
            (name.as_str(), off)
        })
    }

    pub fn get_profile(&self, name: &str) -> Option<ProfileStats> {
        // #364 read-side alias: when BlockCounts (or BlockCountsAndTiming)
        // is enabled the per-function entry counter is NOT stored under
        // `name`; instead the entry block counter under `{name}::block0`
        // serves both purposes. Fall back to that alias so the stable
        // `get_profile(name)` API keeps working regardless of which
        // profile mode compiled the function.
        if let Some(counter) = self.counters.get(name) {
            return Some(ProfileStats {
                call_count: counter.load(Ordering::Relaxed),
            });
        }
        let alias = format!("{}::block0", name);
        if let Some(counter) = self.counters.get(&alias) {
            return Some(ProfileStats {
                call_count: counter.load(Ordering::Relaxed),
            });
        }
        // Phase 3 (BlockCountsAndTiming): the entry counter lives inside
        // the TimingCell, not the `counters` map.
        self.timing_cells.get(&alias).map(|cell| ProfileStats {
            call_count: cell.count.load(Ordering::Relaxed),
        })
    }

    /// Returns the function-entry counter for `name`, or `None` if `name`
    /// was not compiled with entry counters enabled.
    ///
    /// Equivalent to `self.get_profile(name).map(|s| s.call_count)`; provided
    /// as the stable public API per issue #478.
    pub fn entry_count(&self, name: &str) -> Option<u64> {
        self.get_profile(name).map(|stats| stats.call_count)
    }

    /// Returns the per-basic-block counter value for the given function and
    /// block id, or `None` if `name` was not compiled with
    /// [`ProfileHookMode::BlockCounts`] / [`ProfileHookMode::BlockCountsAndTiming`]
    /// or `block_id` is not a valid block of that function.
    ///
    /// The key format (`"{name}::block{block_id.0}"`) is the stable public
    /// API introduced for issue #364. Prover harnesses and tla2 callers
    /// should use this accessor rather than reaching into `counters`-style
    /// internals.
    ///
    /// Under [`ProfileHookMode::BlockCountsAndTiming`] the same count value
    /// is available; this accessor looks in both the plain-counter and the
    /// timing-cell tables so the read-side API is uniform across modes.
    pub fn block_count(&self, name: &str, block_id: llvm2_ir::types::BlockId) -> Option<u64> {
        let key = format!("{}::block{}", name, block_id.0);
        if let Some(counter) = self.counters.get(&key) {
            return Some(counter.load(Ordering::Relaxed));
        }
        self.timing_cells
            .get(&key)
            .map(|cell| cell.count.load(Ordering::Relaxed))
    }

    /// Iterate `(block_id_value, count)` pairs for every block-level counter
    /// registered under `name`. Yields nothing if `name` was not compiled
    /// with [`ProfileHookMode::BlockCounts`] or
    /// [`ProfileHookMode::BlockCountsAndTiming`].
    ///
    /// Returned in an unspecified order. Callers that need deterministic
    /// ordering should collect + sort by the first tuple element.
    pub fn block_counts(&self, name: &str) -> Vec<(u32, u64)> {
        let prefix = format!("{}::block", name);
        let from_counters = self
            .counters
            .iter()
            .filter_map(|(key, counter)| {
                let rest = key.strip_prefix(&prefix)?;
                let bid: u32 = rest.parse().ok()?;
                Some((bid, counter.load(Ordering::Relaxed)))
            });
        let from_timing = self.timing_cells.iter().filter_map(|(key, cell)| {
            let rest = key.strip_prefix(&prefix)?;
            let bid: u32 = rest.parse().ok()?;
            Some((bid, cell.count.load(Ordering::Relaxed)))
        });
        from_counters.chain(from_timing).collect()
    }

    /// Returns `(count, total_cycles)` for the given function's basic
    /// block, or `None` if `name` was not compiled with
    /// [`ProfileHookMode::BlockCountsAndTiming`] (#364 Phase 3) or
    /// `block_id` is not a valid block of that function.
    ///
    /// `total_cycles` is the accumulated `CNTVCT_EL0` delta between this
    /// block's entry and the next block's entry (on any thread, since the
    /// attribution state is buffer-wide). See [`TimingState`] for the
    /// attribution semantics. The first block entered under a buffer
    /// contributes 0 cycles because there is no preceding entry to
    /// attribute from.
    pub fn block_timing(
        &self,
        name: &str,
        block_id: llvm2_ir::types::BlockId,
    ) -> Option<(u64, u64)> {
        let key = format!("{}::block{}", name, block_id.0);
        self.timing_cells.get(&key).map(|cell| {
            (
                cell.count.load(Ordering::Relaxed),
                cell.total_cycles.load(Ordering::Relaxed),
            )
        })
    }

    /// Iterate `(block_id_value, count, total_cycles)` tuples for every
    /// timing cell registered under `name`. Yields nothing if `name` was
    /// not compiled with [`ProfileHookMode::BlockCountsAndTiming`].
    ///
    /// Returned in an unspecified order. Callers that need deterministic
    /// ordering should collect + sort by the first tuple element.
    pub fn block_timings(&self, name: &str) -> Vec<(u32, u64, u64)> {
        let prefix = format!("{}::block", name);
        self.timing_cells
            .iter()
            .filter_map(|(key, cell)| {
                let rest = key.strip_prefix(&prefix)?;
                let bid: u32 = rest.parse().ok()?;
                Some((
                    bid,
                    cell.count.load(Ordering::Relaxed),
                    cell.total_cycles.load(Ordering::Relaxed),
                ))
            })
            .collect()
    }

    /// Reset the call counter for `name` to 0. Returns `true` if `name` was a
    /// profiled function (counter existed), `false` otherwise. Callers that
    /// need the pre-reset count should call `get_profile` first — the reset
    /// itself is destructive and does not return the old value.
    ///
    /// Uses `Ordering::Relaxed` to match the trampoline increment side. Safe
    /// to call while the function is executing on other threads; the observed
    /// counter on those threads may miss increments that happen across the
    /// reset boundary, which is the documented relaxed-atomic behaviour.
    pub fn reset_profile(&self, name: &str) -> bool {
        if let Some(counter) = self.counters.get(name) {
            counter.store(0, Ordering::Relaxed);
            true
        } else {
            false
        }
    }

    /// Reset the entry counter for `name` to zero. Returns `true` iff `name`
    /// had a counter. Equivalent to [`Self::reset_profile`] and kept separate
    /// so the #478 public API is self-contained.
    pub fn reset_entry_count(&self, name: &str) -> bool {
        self.reset_profile(name)
    }

    /// Reset every profiled function's counter to 0. Returns the number of
    /// counters that were reset.
    pub fn reset_all_profiles(&self) -> usize {
        for counter in self.counters.values() {
            counter.store(0, Ordering::Relaxed);
        }
        self.counters.len()
    }

    pub fn profiles(&self) -> impl Iterator<Item = (&str, ProfileStats)> + '_ {
        self.counters.iter().map(|(name, counter)| {
            (
                name.as_str(),
                ProfileStats {
                    call_count: counter.load(Ordering::Relaxed),
                },
            )
        })
    }

    /// Snapshot every `(name, count)` pair. Produces an owned `Vec` so callers
    /// can drop the buffer's borrow between samples.
    pub fn entry_counts(&self) -> Vec<(String, u64)> {
        self.canonical_symbols
            .iter()
            .filter_map(|name| {
                self.counters
                    .get(name)
                    .map(|counter| (name.clone(), counter.load(Ordering::Relaxed)))
            })
            .collect()
    }

    // --- Proof certificates (issue #348) -------------------------------------

    /// Return the proof certificate for the named function, if one was
    /// generated. A certificate is generated only when
    /// [`crate::jit::JitConfig::verify`] was true at compile time and the
    /// `verify` feature is enabled in the build.
    ///
    /// Callers such as tla2 use this to assert that a JIT'd function has
    /// been formally checked against its tMIR input. Example:
    ///
    /// ```no_run
    /// # use llvm2_codegen::jit::{JitCompiler, JitConfig};
    /// # use std::collections::HashMap;
    /// let jit = JitCompiler::new(JitConfig { verify: true, ..Default::default() });
    /// # let functions = vec![];
    /// let buf = jit.compile_raw(&functions, &HashMap::new()).unwrap();
    /// if let Some(cert) = buf.certificate("add") {
    ///     assert!(cert.is_verified());
    ///     assert!(cert.replay_check());
    /// }
    /// ```
    pub fn certificate(&self, name: &str) -> Option<&crate::jit_cert::JitCertificate> {
        self.certificates.get(name)
    }

    /// Iterate over every `(function_name, certificate)` pair attached to
    /// this buffer. Empty when verification was disabled.
    pub fn certificates(&self) -> impl Iterator<Item = (&str, &crate::jit_cert::JitCertificate)> {
        self.certificates.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Returns true iff every attached certificate reports
    /// [`crate::jit_cert::JitCertificate::is_verified`]. Returns `true`
    /// vacuously when there are no certificates (e.g. verification off).
    pub fn all_verified(&self) -> bool {
        self.certificates.values().all(|c| c.is_verified())
    }

    /// Export every attached certificate as a single JSON object mapping
    /// function name → certificate JSON. Intended for cross-system proof
    /// composition (see `designs/2026-04-16-proof-certificate-chain.md`).
    pub fn export_proofs(&self) -> String {
        let mut out = String::from("{\n  \"functions\": {");
        for (i, (name, cert)) in self.certificates.iter().enumerate() {
            if i > 0 {
                out.push_str(",");
            }
            out.push_str(&format!(
                "\n    \"{}\": ",
                crate::jit_cert::escape_for_export(name)
            ));
            out.push_str(&cert.to_json());
        }
        out.push_str("\n  }\n}");
        out
    }
}

impl Drop for ExecutableBuffer {
    fn drop(&mut self) {
        // ORDER MATTERS (#494): `munmap` must run BEFORE the counter /
        // timing-cell / timing-state boxes are dropped. This runs first
        // (`Drop::drop` is invoked before the compiler-generated field
        // drops), and the counter/timing fields are declared AFTER
        // `memory` in `ExecutableBuffer` so they drop after this returns.
        //
        // Consequence: once `munmap` returns, any in-flight JIT call is
        // already a use-after-free on the text pages themselves — there
        // is no window in which the trampoline can observe a freed
        // counter while the text is still mapped. See the module-level
        // "Profile counter & timing-cell lifetime" section.
        if !self.memory.is_null() {
            unsafe {
                sys::munmap(self.memory, self.len);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Architecture-specific fixup patching and veneer stubs
// ---------------------------------------------------------------------------

/// Patch a branch/call fixup at `offset` to reach `target`.
/// Dispatches to the appropriate architecture-specific implementation.
#[cfg(target_arch = "aarch64")]
fn patch_fixup(code: &mut [u8], offset: u32, target: u64) -> Result<(), JitError> {
    patch_branch26(code, offset, target)
}

#[cfg(target_arch = "x86_64")]
fn patch_fixup(code: &mut [u8], offset: u32, target: u64) -> Result<(), JitError> {
    patch_rel32(code, offset, target)
}

/// Emit a veneer trampoline stub that loads a 64-bit address and jumps to it.
/// The absolute address slot is filled in later.
#[cfg(target_arch = "aarch64")]
fn emit_veneer_stub(code: &mut Vec<u8>) {
    // LDR X16, [PC, #8] (0x58000050) ; BR X16 (0xD61F0200) ; .quad <addr>
    // Total: 16 bytes
    code.extend_from_slice(&0x5800_0050u32.to_le_bytes());
    code.extend_from_slice(&0xD61F_0200u32.to_le_bytes());
    code.extend_from_slice(&[0u8; 8]);
}

#[cfg(target_arch = "aarch64")]
fn emit_profile_trampoline_aarch64(code: &mut Vec<u8>) -> usize {
    let literal_slot_offset = code.len() + 20;
    // Load the counter pointer, increment it, then branch over the literal
    // slot into the real function body.
    code.extend_from_slice(&0x5800_00B0u32.to_le_bytes());
    code.extend_from_slice(&0xF940_0211u32.to_le_bytes());
    code.extend_from_slice(&0x9100_0631u32.to_le_bytes());
    code.extend_from_slice(&0xF900_0211u32.to_le_bytes());
    code.extend_from_slice(&0x1400_0003u32.to_le_bytes());
    code.extend_from_slice(&[0u8; 8]);
    literal_slot_offset
}

/// Size in bytes of the AArch64 profile trampoline emitted by
/// [`emit_profile_trampoline_aarch64`]. Kept as a named constant so the
/// block-splicing logic and the byte-layout unit test stay in sync.
#[cfg(target_arch = "aarch64")]
const AARCH64_PROFILE_TRAMPOLINE_BYTES: usize = 28;

/// Size in bytes of the AArch64 `BlockCountsAndTiming` trampoline
/// emitted by [`emit_profile_trampoline_with_timing_aarch64`]
/// (issue #364, Phase 3).
///
/// Layout (byte offsets within a single trampoline):
///
/// ```text
///   0  STP X9, X10,  [SP, #-16]!      ; save caller-saved temps
///   4  STP X11, X12, [SP, #-16]!      ;
///   8  LDR X16, [PC, #lit_counter]    ; X16 = &BlockTimingCell
///  12  LDR X17, [X16]                 ; count++
///  16  ADD X17, X17, #1               ;
///  20  STR X17, [X16]                 ;
///  24  MRS X17, CNTVCT_EL0            ; X17 = now
///  28  ADD X11, X16, #8               ; X11 = &cell.total_cycles
///  32  LDR X9,  [PC, #lit_tstate]     ; X9 = &TimingState
///  36  LDR X10, [X9]                  ; X10 = prev_ts
///  40  CBZ X10, skip_attrib           ; +24 → off 64
///  44    SUB X12, X17, X10            ; X12 = now - prev_ts
///  48    LDR X16, [X9, #8]            ; X16 = prev_accum_ptr
///  52    LDR X10, [X16]               ; X10 = *prev_accum
///  56    ADD X10, X10, X12            ; X10 += delta
///  60    STR X10, [X16]               ; *prev_accum = X10
///  64  skip_attrib: STR X17, [X9]     ; prev_ts = now
///  68  STR X11, [X9, #8]              ; prev_accum_ptr = &cell.total_cycles
///  72  LDP X11, X12, [SP], #16
///  76  LDP X9,  X10, [SP], #16
///  80  B   over_literals              ; +20 → off 100
///  84  .quad lit_counter              ; patched to &BlockTimingCell
///  92  .quad lit_tstate               ; patched to &TimingState
/// 100  (end; block body follows)
/// ```
///
/// The scratch-saving prologue uses `{X9, X10, X11, X12}` — all
/// AAPCS64 caller-saved temporaries — and leaves `{X0..X8}` untouched
/// so argument registers are preserved across every block entry,
/// including the entry block (which doubles as the function entry).
/// `X16`/`X17` are IP0/IP1 and already ABI-free scratch; the trampoline
/// clobbers them without saving them. `X18` (Darwin platform register)
/// is intentionally NEVER touched.
///
/// The literal pool lives inside the trampoline so each block's
/// trampoline has its own two literals (counter cell pointer and
/// timing-state pointer) — no cross-trampoline reference chains to
/// complicate re-patching when trampolines are spliced in.
#[cfg(target_arch = "aarch64")]
const AARCH64_PROFILE_TRAMPOLINE_TIMING_BYTES: usize = 100;

/// Emit the AArch64 `BlockCountsAndTiming` trampoline into `code`.
/// Returns `(counter_cell_literal_offset, timing_state_literal_offset)`
/// — both absolute offsets into `code` at the time of return, suitable
/// for pushing onto the compile_raw patch-site lists.
///
/// See [`AARCH64_PROFILE_TRAMPOLINE_TIMING_BYTES`] for the full byte
/// layout, register plan, and ABI notes.
#[cfg(target_arch = "aarch64")]
fn emit_profile_trampoline_with_timing_aarch64(code: &mut Vec<u8>) -> (usize, usize) {
    let start = code.len();
    // Prolog: save X9, X10, X11, X12 (caller-saved temporaries that
    // the attribution body clobbers beyond the ABI-free X16/X17).
    code.extend_from_slice(&0xA9BF_2BE9u32.to_le_bytes()); // STP X9, X10, [SP, #-16]!
    code.extend_from_slice(&0xA9BF_33EBu32.to_le_bytes()); // STP X11, X12, [SP, #-16]!

    // --- count increment ---
    // LDR X16, [PC, #76] → literal at offset +76 from this instruction
    // (instr at start+8, literal at start+84, delta = 76).
    code.extend_from_slice(&0x5800_0270u32.to_le_bytes()); // LDR X16, [PC, #76]
    code.extend_from_slice(&0xF940_0211u32.to_le_bytes()); // LDR X17, [X16]
    code.extend_from_slice(&0x9100_0631u32.to_le_bytes()); // ADD X17, X17, #1
    code.extend_from_slice(&0xF900_0211u32.to_le_bytes()); // STR X17, [X16]

    // --- timing capture + cross-block attribution ---
    code.extend_from_slice(&0xD53B_E051u32.to_le_bytes()); // MRS X17, CNTVCT_EL0
    code.extend_from_slice(&0x9100_220Bu32.to_le_bytes()); // ADD X11, X16, #8
    // LDR X9, [PC, #60] → literal at offset +60 from this instruction
    // (instr at start+32, literal at start+92, delta = 60).
    code.extend_from_slice(&0x5800_01E9u32.to_le_bytes()); // LDR X9, [PC, #60]
    code.extend_from_slice(&0xF940_012Au32.to_le_bytes()); // LDR X10, [X9]
    code.extend_from_slice(&0xB400_00CAu32.to_le_bytes()); // CBZ X10, +24 (skip_attrib)
    code.extend_from_slice(&0xCB0A_022Cu32.to_le_bytes()); // SUB X12, X17, X10
    code.extend_from_slice(&0xF940_0530u32.to_le_bytes()); // LDR X16, [X9, #8]
    code.extend_from_slice(&0xF940_020Au32.to_le_bytes()); // LDR X10, [X16]
    code.extend_from_slice(&0x8B0C_014Au32.to_le_bytes()); // ADD X10, X10, X12
    code.extend_from_slice(&0xF900_020Au32.to_le_bytes()); // STR X10, [X16]
    // skip_attrib:
    code.extend_from_slice(&0xF900_0131u32.to_le_bytes()); // STR X17, [X9]      ; prev_ts = now
    code.extend_from_slice(&0xF900_052Bu32.to_le_bytes()); // STR X11, [X9, #8]  ; prev_accum = &cell.cycles

    // --- epilogue: restore X9, X10, X11, X12 ---
    code.extend_from_slice(&0xA8C1_33EBu32.to_le_bytes()); // LDP X11, X12, [SP], #16
    code.extend_from_slice(&0xA8C1_2BE9u32.to_le_bytes()); // LDP X9, X10,  [SP], #16

    // --- branch over the 16-byte literal pool into the block body ---
    code.extend_from_slice(&0x1400_0005u32.to_le_bytes()); // B +20 (over 16B literals + 4B padding-past-self)

    // Literal pool (patched by compile_raw_inner):
    let lit_counter_offset = code.len();
    code.extend_from_slice(&[0u8; 8]); // .quad <&BlockTimingCell>
    let lit_tstate_offset = code.len();
    code.extend_from_slice(&[0u8; 8]); // .quad <&TimingState>

    debug_assert_eq!(code.len() - start, AARCH64_PROFILE_TRAMPOLINE_TIMING_BYTES);
    debug_assert_eq!(lit_counter_offset - start, 84);
    debug_assert_eq!(lit_tstate_offset - start, 92);

    (lit_counter_offset, lit_tstate_offset)
}

/// Size in bytes of the x86-64 profile trampoline emitted by
/// [`emit_profile_trampoline_x86_64`].
#[cfg(target_arch = "x86_64")]
const X86_64_PROFILE_TRAMPOLINE_BYTES: usize = 16;

/// Splice an AArch64 profile trampoline in front of every basic block of
/// an already-encoded function body.
///
/// Design (issue #364, `ProfileHookMode::BlockCounts`):
/// - The entry block's trampoline doubles as the function-entry trampoline,
///   so a block-profiled function does not need a separate function-entry
///   increment — counting the entry block already counts the call.
/// - Each block gets its own counter, keyed externally by
///   `format!("{}::block{}", func.name, block_id.0)` so the canonical
///   function-entry APIs (`get_profile(name)`) continue to work for the
///   whole function via `block0`.
/// - Intra-function PC-relative branches (`B`, `B.cond`, `CBZ`, `CBNZ`,
///   `TBZ`, `TBNZ`) are re-patched so their displacements still land on
///   the intended target block's new location. After splicing, branch
///   targets that were block starts now point at the *trampoline* for
///   that block, which is exactly what we want — the counter has to fire
///   every time control reaches the block, including on back-edges.
/// - Jump tables (issue #490): `encode_function_with_fixups_and_blocks`
///   appends one 32-bit-entry table per ADR-with-`JumpTableIndex` operand
///   after the function body. The ADR's imm21 was patched to `(table -
///   adr)`, and each entry was written as `(target_block - table_base)`.
///   After splicing, all three terms shift by different amounts, so the
///   splice must (a) strip the original jump-table tail, (b) re-patch
///   each ADR imm21 against the new layout, and (c) regenerate each
///   table entry against post-splice block positions. See issue #490.
/// - The trampoline size (`AARCH64_PROFILE_TRAMPOLINE_BYTES` = 28 bytes =
///   7 instructions) is a multiple of 4, so all branch encodings remain
///   4-byte-aligned.
///
/// Returns the spliced bytes and the list of `(block_id,
/// literal_slot_offset)` patch sites within the spliced bytes.
/// `fixups` is modified in place: each fixup's `offset` is shifted by the
/// cumulative trampoline bytes preceding the fixup's original byte
/// position, so external fixups still index the correct branch
/// instruction in the spliced output.
#[cfg(target_arch = "aarch64")]
fn splice_block_trampolines_aarch64(
    func: &IrMachFunction,
    body_bytes: &[u8],
    block_byte_offsets: &HashMap<BlockId, u32>,
    fixups: &mut [Fixup],
) -> Result<(Vec<u8>, Vec<(BlockId, usize)>), JitError> {
    use llvm2_ir::inst::AArch64Opcode;

    let tramp = AARCH64_PROFILE_TRAMPOLINE_BYTES;

    // Block layout order and per-block shift amounts. Block at layout
    // position `k` (0-indexed) has `(k+1)*tramp` bytes of trampoline
    // inserted at-or-before its first original byte: one trampoline each
    // for block_order[0..=k].
    let mut block_layout_idx: HashMap<BlockId, usize> = HashMap::new();
    for (idx, &bid) in func.block_order.iter().enumerate() {
        block_layout_idx.insert(bid, idx);
    }

    // Compute the new (post-splice) byte offset of each block's trampoline
    // start.
    let mut block_new_trampoline_start: HashMap<BlockId, usize> = HashMap::new();
    for (k, &bid) in func.block_order.iter().enumerate() {
        let orig = *block_byte_offsets.get(&bid).ok_or_else(|| {
            JitError::Pipeline(PipelineError::Encoding(format!(
                "block {:?} missing byte offset for trampoline splice",
                bid
            )))
        })? as usize;
        let tramp_start = orig + k * tramp;
        block_new_trampoline_start.insert(bid, tramp_start);
    }

    // Collect ADR-for-jump-table sites (issue #490). We must replay the
    // exact walk used by `encode_function_with_fixups_and_blocks` so the
    // ordering (and therefore which appended table a given ADR points at)
    // matches the encoder. Each entry records:
    //   - `orig_adr_byte`: byte offset in `body_bytes` of the 4-byte ADR
    //   - `source_layout_idx`: layout position of the containing block
    //   - `jt_idx`: index into `func.jump_tables`
    // Also compute `insts_end_in_body` = the byte offset in `body_bytes`
    // that separates instruction bytes from the appended jump-table tail.
    // Since the encoder writes exactly 4 bytes per non-pseudo instruction,
    // this equals (start of last block) + 4 * non_pseudo_count(last
    // block).
    let mut jt_adr_sites: Vec<(usize, usize, u32)> = Vec::new();
    for (layout_idx, &bid) in func.block_order.iter().enumerate() {
        let block = func.block(bid);
        let mut cur_byte = *block_byte_offsets.get(&bid).unwrap() as usize;
        for &inst_id in &block.insts {
            let inst = func.inst(inst_id);
            if inst.is_pseudo() {
                continue;
            }
            if inst.opcode == AArch64Opcode::Adr {
                if let Some(jt_idx) = inst.operands.get(1).and_then(|op| op.as_jump_table_index()) {
                    jt_adr_sites.push((cur_byte, layout_idx, jt_idx));
                }
            }
            cur_byte += 4;
        }
    }
    let insts_end_in_body = {
        let last_bid = *func.block_order.last().ok_or_else(|| {
            JitError::Pipeline(PipelineError::Encoding(
                "block-splice: empty block_order".to_string(),
            ))
        })?;
        let last_block = func.block(last_bid);
        let last_non_pseudo = last_block
            .insts
            .iter()
            .filter(|&&id| !func.inst(id).is_pseudo())
            .count();
        *block_byte_offsets.get(&last_bid).unwrap() as usize + last_non_pseudo * 4
    };
    if insts_end_in_body > body_bytes.len() {
        return Err(JitError::Pipeline(PipelineError::Encoding(format!(
            "block-splice: computed insts_end_in_body {} exceeds body_bytes.len() {}",
            insts_end_in_body,
            body_bytes.len()
        ))));
    }
    // Sanity: if no ADR->jump-table sites were found, the jump-table tail
    // must be empty. If it's non-empty without any ADR pointing at it, we
    // cannot safely reconstruct the tail, so bail out explicitly rather
    // than silently dropping bytes.
    if jt_adr_sites.is_empty() && insts_end_in_body < body_bytes.len() {
        return Err(JitError::Pipeline(PipelineError::Encoding(format!(
            "block-splice: body_bytes has {} bytes of unexpected tail past instruction end \
             without any Adr(JumpTableIndex) site",
            body_bytes.len() - insts_end_in_body
        ))));
    }
    // Range-check every referenced jt_idx BEFORE we start writing `out`.
    for &(_, _, jt_idx) in &jt_adr_sites {
        if (jt_idx as usize) >= func.jump_tables.len() {
            return Err(JitError::Pipeline(PipelineError::Encoding(format!(
                "block-splice: Adr references jump table index {} but func has only {} tables",
                jt_idx,
                func.jump_tables.len()
            ))));
        }
    }

    // Build the spliced output by walking blocks in layout order,
    // inserting one trampoline then the original block bytes.
    let mut out = Vec::with_capacity(body_bytes.len() + func.block_order.len() * tramp);
    let mut tramp_sites: Vec<(BlockId, usize)> = Vec::with_capacity(func.block_order.len());

    for (k, &bid) in func.block_order.iter().enumerate() {
        let block_orig_start = *block_byte_offsets.get(&bid).unwrap() as usize;
        let block_orig_end = if k + 1 < func.block_order.len() {
            *block_byte_offsets.get(&func.block_order[k + 1]).unwrap() as usize
        } else {
            // Last block: stop at the end of instruction bytes so the
            // appended jump-table tail is NOT copied into the spliced
            // output. The tables are regenerated below against the
            // post-splice layout. (issue #490)
            insts_end_in_body
        };
        // Emit trampoline for this block.
        let literal_slot_offset = emit_profile_trampoline_aarch64(&mut out);
        tramp_sites.push((bid, literal_slot_offset));
        // Sanity: the trampoline just emitted begins at the computed
        // post-splice offset.
        debug_assert_eq!(
            out.len() - tramp,
            *block_new_trampoline_start.get(&bid).unwrap()
        );
        // Append the block's original bytes.
        out.extend_from_slice(&body_bytes[block_orig_start..block_orig_end]);
    }

    // Re-patch intra-function PC-relative branches so displacements still
    // resolve to the intended target block. We iterate instructions in
    // original layout order so we can compute the original byte offset of
    // each branch instruction.
    for &bid in &func.block_order {
        let block = func.block(bid);
        let mut orig_inst_byte = *block_byte_offsets.get(&bid).unwrap() as usize;
        for &inst_id in &block.insts {
            let inst = func.inst(inst_id);
            if inst.is_pseudo() {
                continue;
            }
            let opcode = inst.opcode;
            let is_branch_with_symbol = matches!(
                opcode,
                AArch64Opcode::B | AArch64Opcode::Bl | AArch64Opcode::BL
            ) && inst.operands.first().map_or(false, |op| op.is_symbol());

            // Intra-function PC-relative branches we need to re-patch. BL
            // with a symbol target is NOT re-patched here — that's an
            // external fixup resolved later.
            let is_intra_branch = matches!(
                opcode,
                AArch64Opcode::B
                    | AArch64Opcode::BCond
                    | AArch64Opcode::Cbz
                    | AArch64Opcode::Cbnz
                    | AArch64Opcode::Tbz
                    | AArch64Opcode::Tbnz
            ) && !is_branch_with_symbol;

            if is_intra_branch {
                // Locate the new source byte in `out`.
                let source_layout_idx = *block_layout_idx.get(&bid).unwrap();
                let new_source = orig_inst_byte + (source_layout_idx + 1) * tramp;

                // Decode the existing 4-byte instruction from `out`, get
                // its original (baked-in) imm field, compute the original
                // target byte offset, find the target block, and re-encode
                // with the shifted displacement.
                let existing = u32::from_le_bytes([
                    out[new_source],
                    out[new_source + 1],
                    out[new_source + 2],
                    out[new_source + 3],
                ]);

                let (imm_bits, imm_shift, imm_mask, sign_bits) = match opcode {
                    AArch64Opcode::B | AArch64Opcode::Bl | AArch64Opcode::BL => {
                        (26u32, 0u32, 0x03FF_FFFFu32, 25u32)
                    }
                    AArch64Opcode::BCond | AArch64Opcode::Cbz | AArch64Opcode::Cbnz => {
                        (19, 5, 0x0007_FFFF, 18)
                    }
                    AArch64Opcode::Tbz | AArch64Opcode::Tbnz => (14, 5, 0x0000_3FFF, 13),
                    _ => unreachable!(),
                };
                let raw_imm = (existing >> imm_shift) & imm_mask;
                // Sign-extend from `imm_bits` bits.
                let sign = (raw_imm >> sign_bits) & 1;
                let signed_imm = if sign == 1 {
                    (raw_imm as i64) | !((1i64 << imm_bits) - 1)
                } else {
                    raw_imm as i64
                };
                let original_target_byte = (orig_inst_byte as i64) + signed_imm * 4;
                if original_target_byte < 0 {
                    return Err(JitError::Pipeline(PipelineError::Encoding(format!(
                        "block-splice: negative original branch target {}",
                        original_target_byte
                    ))));
                }
                let original_target_byte = original_target_byte as u32;

                // Look up which block starts at `original_target_byte`.
                // By the resolve_branches invariant, every intra-function
                // branch's target is a block start.
                let target_bid = block_byte_offsets
                    .iter()
                    .find(|&(_, &bo)| bo == original_target_byte)
                    .map(|(bid, _)| *bid)
                    .ok_or_else(|| {
                        JitError::Pipeline(PipelineError::Encoding(format!(
                            "block-splice: branch at original byte {} targets {} which is not a block start",
                            orig_inst_byte, original_target_byte
                        )))
                    })?;
                let new_target = *block_new_trampoline_start.get(&target_bid).unwrap();

                // Compute new byte-distance and encode.
                let new_dist_bytes = new_target as i64 - new_source as i64;
                if new_dist_bytes % 4 != 0 {
                    return Err(JitError::Pipeline(PipelineError::Encoding(format!(
                        "block-splice: non-4-byte-aligned branch distance {}",
                        new_dist_bytes
                    ))));
                }
                let new_inst_units = new_dist_bytes / 4;
                // Range-check.
                let range = 1i64 << (imm_bits - 1);
                if new_inst_units < -range || new_inst_units >= range {
                    return Err(JitError::Pipeline(PipelineError::Encoding(format!(
                        "block-splice: branch displacement {} out of {}-bit range after \
                         trampoline insertion",
                        new_inst_units, imm_bits
                    ))));
                }
                let new_imm = (new_inst_units as u32) & imm_mask;
                let cleared = existing & !(imm_mask << imm_shift);
                let rewritten = cleared | (new_imm << imm_shift);
                out[new_source..new_source + 4].copy_from_slice(&rewritten.to_le_bytes());
            }

            orig_inst_byte += 4;
        }
    }

    // Shift external fixups: each fixup's `offset` pointed into the
    // original `body_bytes`. In the new `out` it must shift by
    // `(source_layout_idx + 1) * tramp` where `source_layout_idx` is the
    // layout position of the block containing the fixup.
    let mut layout_starts: Vec<(usize, usize)> = func
        .block_order
        .iter()
        .enumerate()
        .map(|(idx, bid)| (*block_byte_offsets.get(bid).unwrap() as usize, idx))
        .collect();
    layout_starts.sort_by_key(|&(off, _)| off);

    for fx in fixups.iter_mut() {
        let off = fx.offset as usize;
        // Find the largest layout_start.0 <= off; that defines which block
        // contains this fixup.
        let mut layout_idx = 0usize;
        for (start, idx) in layout_starts.iter() {
            if *start <= off {
                layout_idx = *idx;
            } else {
                break;
            }
        }
        let shift = (layout_idx + 1) * tramp;
        fx.offset = (off + shift) as u32;
    }

    // Regenerate jump tables against the post-splice layout (issue #490).
    //
    // Replay the pipeline.rs emission walk: for each ADR->jump-table site
    // (in the same deterministic order the encoder used), append one
    // table's worth of bytes at the current end of `out`, then re-patch
    // the ADR's imm21 to point at that new table base.
    //
    // Each table entry is a 32-bit signed delta
    // `(target_block_new_offset - new_table_base)` where
    // `target_block_new_offset` is the post-splice start of the TARGET
    // BLOCK'S TRAMPOLINE. Landing on the trampoline (not the first real
    // instruction) matches the convention used for regular branch
    // re-patching so every jump-table-dispatched case increments the
    // target block's counter.
    for (orig_adr_byte, source_layout_idx, jt_idx) in jt_adr_sites {
        let new_adr = orig_adr_byte + (source_layout_idx + 1) * tramp;
        if new_adr + 4 > out.len() {
            return Err(JitError::Pipeline(PipelineError::Encoding(format!(
                "block-splice: ADR site at post-splice offset {} exceeds out buffer length {}",
                new_adr,
                out.len()
            ))));
        }

        let new_table_base = out.len();
        let pc_relative = new_table_base as i64 - new_adr as i64;
        if !(-(1i64 << 20)..(1i64 << 20)).contains(&pc_relative) {
            return Err(JitError::Pipeline(PipelineError::Encoding(format!(
                "block-splice: ADR->jump-table offset {} does not fit in imm21",
                pc_relative
            ))));
        }

        // Re-encode the ADR, preserving Rd from the placeholder bits[4:0].
        let placeholder_word = u32::from_le_bytes([
            out[new_adr],
            out[new_adr + 1],
            out[new_adr + 2],
            out[new_adr + 3],
        ]);
        let rd = (placeholder_word & 0x1F) as u8;
        let patched = crate::aarch64::encoding_mem::encode_adr(pc_relative as i32, rd)
            .map_err(|e| JitError::Pipeline(PipelineError::Encoding(e.to_string())))?;
        out[new_adr..new_adr + 4].copy_from_slice(&patched.to_le_bytes());

        // Append one table copy. (If two ADRs share the same jt_idx the
        // pipeline emits two separate tables; we mirror that exactly.)
        let jt = &func.jump_tables[jt_idx as usize];
        for target in &jt.targets {
            let new_target = *block_new_trampoline_start.get(target).ok_or_else(|| {
                JitError::Pipeline(PipelineError::Encoding(format!(
                    "block-splice: jump-table target {:?} has no post-splice offset",
                    target
                )))
            })?;
            let entry: i32 = (new_target as i64 - new_table_base as i64) as i32;
            out.extend_from_slice(&entry.to_le_bytes());
        }
    }

    Ok((out, tramp_sites))
}

/// Splice an AArch64 `BlockCountsAndTiming` trampoline in front of every
/// basic block of an already-encoded function body (issue #364, Phase 3).
///
/// This is the timing-aware sibling of [`splice_block_trampolines_aarch64`].
/// It follows the same re-patching discipline for intra-function PC-relative
/// branches but:
/// - Emits 100-byte [`emit_profile_trampoline_with_timing_aarch64`] trampolines
///   instead of 28-byte plain-counter ones.
/// - Returns two patch-site offsets per block: one for the
///   `&BlockTimingCell` literal and one for the `&TimingState` literal.
///
/// The trampoline size is a multiple of 4, so branch encoding alignment
/// remains valid. The 14-bit range of `TBZ/TBNZ` allows roughly
/// `(2^13 * 4) / 100 ≈ 320` blocks per function before displacement
/// overflow; that is large enough for any human-written function and is
/// documented as a Phase 3 limitation in #364.
///
/// Returns the spliced bytes and a list of
/// `(block_id, counter_literal_offset, tstate_literal_offset)` patch sites
/// pointing into the spliced bytes. `fixups` is shifted in place.
#[cfg(target_arch = "aarch64")]
fn splice_block_trampolines_with_timing_aarch64(
    func: &IrMachFunction,
    body_bytes: &[u8],
    block_byte_offsets: &HashMap<BlockId, u32>,
    fixups: &mut [Fixup],
) -> Result<(Vec<u8>, Vec<(BlockId, usize, usize)>), JitError> {
    use llvm2_ir::inst::AArch64Opcode;

    let tramp = AARCH64_PROFILE_TRAMPOLINE_TIMING_BYTES;

    // Block layout order and per-block shift amounts.
    let mut block_layout_idx: HashMap<BlockId, usize> = HashMap::new();
    for (idx, &bid) in func.block_order.iter().enumerate() {
        block_layout_idx.insert(bid, idx);
    }

    // Compute the new (post-splice) byte offset of each block's trampoline
    // start.
    let mut block_new_trampoline_start: HashMap<BlockId, usize> = HashMap::new();
    for (k, &bid) in func.block_order.iter().enumerate() {
        let orig = *block_byte_offsets.get(&bid).ok_or_else(|| {
            JitError::Pipeline(PipelineError::Encoding(format!(
                "block {:?} missing byte offset for timing-trampoline splice",
                bid
            )))
        })? as usize;
        let tramp_start = orig + k * tramp;
        block_new_trampoline_start.insert(bid, tramp_start);
    }

    // Build spliced output.
    let mut out = Vec::with_capacity(body_bytes.len() + func.block_order.len() * tramp);
    let mut tramp_sites: Vec<(BlockId, usize, usize)> =
        Vec::with_capacity(func.block_order.len());

    for (k, &bid) in func.block_order.iter().enumerate() {
        let block_orig_start = *block_byte_offsets.get(&bid).unwrap() as usize;
        let block_orig_end = if k + 1 < func.block_order.len() {
            *block_byte_offsets.get(&func.block_order[k + 1]).unwrap() as usize
        } else {
            body_bytes.len()
        };
        let (lit_counter_offset, lit_tstate_offset) =
            emit_profile_trampoline_with_timing_aarch64(&mut out);
        tramp_sites.push((bid, lit_counter_offset, lit_tstate_offset));
        debug_assert_eq!(
            out.len() - tramp,
            *block_new_trampoline_start.get(&bid).unwrap()
        );
        out.extend_from_slice(&body_bytes[block_orig_start..block_orig_end]);
    }

    // Re-patch intra-function PC-relative branches so displacements still
    // resolve to the intended target block. Identical logic to the
    // plain-counter splicer; only the `tramp` constant differs.
    for &bid in &func.block_order {
        let block = func.block(bid);
        let mut orig_inst_byte = *block_byte_offsets.get(&bid).unwrap() as usize;
        for &inst_id in &block.insts {
            let inst = func.inst(inst_id);
            if inst.is_pseudo() {
                continue;
            }
            let opcode = inst.opcode;
            let is_branch_with_symbol = matches!(
                opcode,
                AArch64Opcode::B | AArch64Opcode::Bl | AArch64Opcode::BL
            ) && inst.operands.first().map_or(false, |op| op.is_symbol());

            let is_intra_branch = matches!(
                opcode,
                AArch64Opcode::B
                    | AArch64Opcode::BCond
                    | AArch64Opcode::Cbz
                    | AArch64Opcode::Cbnz
                    | AArch64Opcode::Tbz
                    | AArch64Opcode::Tbnz
            ) && !is_branch_with_symbol;

            if is_intra_branch {
                let source_layout_idx = *block_layout_idx.get(&bid).unwrap();
                let new_source = orig_inst_byte + (source_layout_idx + 1) * tramp;

                let existing = u32::from_le_bytes([
                    out[new_source],
                    out[new_source + 1],
                    out[new_source + 2],
                    out[new_source + 3],
                ]);

                let (imm_bits, imm_shift, imm_mask, sign_bits) = match opcode {
                    AArch64Opcode::B | AArch64Opcode::Bl | AArch64Opcode::BL => {
                        (26u32, 0u32, 0x03FF_FFFFu32, 25u32)
                    }
                    AArch64Opcode::BCond | AArch64Opcode::Cbz | AArch64Opcode::Cbnz => {
                        (19, 5, 0x0007_FFFF, 18)
                    }
                    AArch64Opcode::Tbz | AArch64Opcode::Tbnz => (14, 5, 0x0000_3FFF, 13),
                    _ => unreachable!(),
                };
                let raw_imm = (existing >> imm_shift) & imm_mask;
                let sign = (raw_imm >> sign_bits) & 1;
                let signed_imm = if sign == 1 {
                    (raw_imm as i64) | !((1i64 << imm_bits) - 1)
                } else {
                    raw_imm as i64
                };
                let original_target_byte = (orig_inst_byte as i64) + signed_imm * 4;
                if original_target_byte < 0 {
                    return Err(JitError::Pipeline(PipelineError::Encoding(format!(
                        "timing-block-splice: negative original branch target {}",
                        original_target_byte
                    ))));
                }
                let original_target_byte = original_target_byte as u32;

                let target_bid = block_byte_offsets
                    .iter()
                    .find(|&(_, &bo)| bo == original_target_byte)
                    .map(|(bid, _)| *bid)
                    .ok_or_else(|| {
                        JitError::Pipeline(PipelineError::Encoding(format!(
                            "timing-block-splice: branch at original byte {} targets {} which is not a block start",
                            orig_inst_byte, original_target_byte
                        )))
                    })?;
                let new_target = *block_new_trampoline_start.get(&target_bid).unwrap();

                let new_dist_bytes = new_target as i64 - new_source as i64;
                if new_dist_bytes % 4 != 0 {
                    return Err(JitError::Pipeline(PipelineError::Encoding(format!(
                        "timing-block-splice: non-4-byte-aligned branch distance {}",
                        new_dist_bytes
                    ))));
                }
                let new_inst_units = new_dist_bytes / 4;
                let range = 1i64 << (imm_bits - 1);
                if new_inst_units < -range || new_inst_units >= range {
                    return Err(JitError::Pipeline(PipelineError::Encoding(format!(
                        "timing-block-splice: branch displacement {} out of {}-bit range after \
                         trampoline insertion (block count × 100-byte trampolines too large)",
                        new_inst_units, imm_bits
                    ))));
                }
                let new_imm = (new_inst_units as u32) & imm_mask;
                let cleared = existing & !(imm_mask << imm_shift);
                let rewritten = cleared | (new_imm << imm_shift);
                out[new_source..new_source + 4].copy_from_slice(&rewritten.to_le_bytes());
            }

            orig_inst_byte += 4;
        }
    }

    // Shift external fixups.
    let mut layout_starts: Vec<(usize, usize)> = func
        .block_order
        .iter()
        .enumerate()
        .map(|(idx, bid)| (*block_byte_offsets.get(bid).unwrap() as usize, idx))
        .collect();
    layout_starts.sort_by_key(|&(off, _)| off);

    for fx in fixups.iter_mut() {
        let off = fx.offset as usize;
        let mut layout_idx = 0usize;
        for (start, idx) in layout_starts.iter() {
            if *start <= off {
                layout_idx = *idx;
            } else {
                break;
            }
        }
        let shift = (layout_idx + 1) * tramp;
        fx.offset = (off + shift) as u32;
    }

    Ok((out, tramp_sites))
}

#[cfg(target_arch = "x86_64")]
fn emit_profile_trampoline_x86_64(code: &mut Vec<u8>) -> usize {
    // Function-entry atomic counter trampoline for x86-64 (#478).
    //
    // Byte layout (16 bytes total):
    //   50              push rax                 ; 1 byte — preserve caller's RAX
    //   48 B8 ii ii ii ii ii ii ii ii
    //                   movabs rax, imm64        ; 10 bytes — heap counter ptr
    //   F0 48 FF 00     lock incq qword ptr [rax]; 4 bytes — atomic increment
    //   58              pop rax                  ; 1 byte — restore caller's RAX
    //
    // RAX must be preserved: the System V AMD64 ABI uses RAX at call sites
    // to pass the number of vector (XMM) registers used when calling
    // variadic functions. Clobbering RAX at function entry would corrupt
    // that convention, so the trampoline wraps the increment in a
    // `push rax` / `pop rax` pair. The sequence is self-balanced (one push,
    // one pop) and preserves stack alignment across the trampoline.
    //
    // Returns the byte offset (within the trampoline's start-of-code slice)
    // of the 8-byte `imm64` field; callers patch it with the live heap
    // counter pointer once the code buffer is mapped.
    code.extend_from_slice(&[0x50]);
    code.extend_from_slice(&[0x48, 0xB8]);
    let imm64_offset = code.len();
    code.extend_from_slice(&[0u8; 8]);
    code.extend_from_slice(&[0xF0, 0x48, 0xFF, 0x00]);
    code.extend_from_slice(&[0x58]);
    imm64_offset
}

#[cfg(target_arch = "x86_64")]
fn emit_veneer_stub(code: &mut Vec<u8>) {
    // JMP [RIP+0]: FF 25 00 00 00 00 ; .quad <addr>
    // Total: 14 bytes. Pad to 16 for alignment.
    code.extend_from_slice(&[0xFF, 0x25, 0x00, 0x00, 0x00, 0x00]); // JMP [RIP+0]
    code.extend_from_slice(&[0u8; 8]); // .quad addr
    code.extend_from_slice(&[0xCC, 0xCC]); // INT3 padding
}

/// Byte offset from veneer start to the embedded absolute address slot.
#[cfg(target_arch = "aarch64")]
const fn veneer_addr_offset() -> usize {
    8 // LDR(4) + BR(4), then 8-byte address
}

#[cfg(target_arch = "x86_64")]
const fn veneer_addr_offset() -> usize {
    6 // FF 25 00 00 00 00 (6 bytes), then 8-byte address
}

/// Patch an AArch64 Branch26 fixup (B/BL instruction, imm26 field).
#[cfg(target_arch = "aarch64")]
fn patch_branch26(code: &mut [u8], offset: u32, target: u64) -> Result<(), JitError> {
    let off = offset as usize;
    if off + 4 > code.len() {
        return Err(JitError::FixupOutOfBounds {
            offset,
            code_len: code.len(),
        });
    }
    let distance = target as i64 - offset as i64;
    if distance < -AARCH64_BRANCH26_MAX || distance >= AARCH64_BRANCH26_MAX {
        return Err(JitError::BranchOutOfRange {
            offset,
            target,
            distance,
        });
    }
    let imm26 = ((distance >> 2) & 0x03FF_FFFF) as u32;
    let existing = u32::from_le_bytes([code[off], code[off + 1], code[off + 2], code[off + 3]]);
    code[off..off + 4].copy_from_slice(&((existing & 0xFC00_0000) | imm26).to_le_bytes());
    Ok(())
}

/// Patch an x86-64 rel32 fixup (CALL/JMP instruction, 4-byte displacement).
///
/// The `offset` is the byte position of the start of the CALL/JMP instruction.
/// For CALL (E8 xx xx xx xx), the displacement is at offset+1 and is relative
/// to the end of the instruction (offset+5). For JMP (E9), same layout.
/// For Jcc (0F 8x xx xx xx xx), displacement at offset+2, end at offset+6.
#[cfg(target_arch = "x86_64")]
fn patch_rel32(code: &mut [u8], offset: u32, target: u64) -> Result<(), JitError> {
    let off = offset as usize;
    // Jcc needs 6 bytes; CALL/JMP need 5. Check the max to avoid OOB panic.
    if off + 6 > code.len() {
        return Err(JitError::FixupOutOfBounds {
            offset,
            code_len: code.len(),
        });
    }
    // Determine instruction length from the opcode to locate the displacement.
    let (disp_off, inst_end) = match code[off] {
        0xE8 | 0xE9 => (off + 1, off + 5), // CALL rel32 / JMP rel32
        0x0F => (off + 2, off + 6),        // Jcc rel32 (0F 80+cc)
        _ => (off + 1, off + 5),           // Default: 5-byte near call/jmp
    };
    let distance = target as i64 - inst_end as i64;
    if distance < i32::MIN as i64 || distance > i32::MAX as i64 {
        return Err(JitError::BranchOutOfRange {
            offset,
            target,
            distance,
        });
    }
    code[disp_off..disp_off + 4].copy_from_slice(&(distance as i32).to_le_bytes());
    Ok(())
}

#[cfg(test)]
mod tests {
    // Test module intentionally exercises the deprecated `get_fn` /
    // `get_fn_ptr` APIs (issue #355) alongside the new bound versions.
    #![allow(deprecated)]
    use super::*;
    #[cfg(target_arch = "aarch64")]
    use llvm2_ir::function::{MachFunction, Signature, Type};
    #[cfg(target_arch = "aarch64")]
    use llvm2_ir::inst::{AArch64Opcode, MachInst};
    #[cfg(target_arch = "aarch64")]
    use llvm2_ir::operand::MachOperand;
    #[cfg(target_arch = "aarch64")]
    use llvm2_ir::regs::X0;

    #[cfg(target_arch = "aarch64")]
    fn build_return_const_named(name: &str) -> MachFunction {
        let sig = Signature::new(vec![], vec![Type::I64]);
        let mut func = MachFunction::new(name.to_string(), sig);
        let entry = func.entry;

        let mov = MachInst::new(
            AArch64Opcode::Movz,
            vec![MachOperand::PReg(X0), MachOperand::Imm(42)],
        );
        let mov_id = func.push_inst(mov);
        func.append_inst(entry, mov_id);

        let ret = MachInst::new(AArch64Opcode::Ret, vec![]);
        let ret_id = func.push_inst(ret);
        func.append_inst(entry, ret_id);

        func
    }

    // -- AArch64 patch tests ---------------------------------------------------

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_patch_branch26_forward() {
        let mut code = vec![0; 20];
        code[0..4].copy_from_slice(&0x9400_0000u32.to_le_bytes());
        patch_branch26(&mut code, 0, 16).unwrap();
        let patched = u32::from_le_bytes([code[0], code[1], code[2], code[3]]);
        assert_eq!(patched, 0x9400_0004);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_patch_branch26_backward() {
        let mut code = vec![0; 20];
        code[8..12].copy_from_slice(&0x1400_0000u32.to_le_bytes());
        patch_branch26(&mut code, 8, 0).unwrap();
        let patched = u32::from_le_bytes([code[8], code[9], code[10], code[11]]);
        assert_eq!(patched, 0x1400_0000 | ((-2i32 as u32) & 0x03FF_FFFF));
    }

    // -- x86-64 patch tests ----------------------------------------------------

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_patch_rel32_call_forward() {
        // CALL rel32 at offset 0, target at offset 20.
        // Instruction is 5 bytes (E8 + 4-byte disp), so disp = 20 - 5 = 15.
        let mut code = vec![0u8; 32];
        code[0] = 0xE8; // CALL opcode
        patch_rel32(&mut code, 0, 20).unwrap();
        let disp = i32::from_le_bytes([code[1], code[2], code[3], code[4]]);
        assert_eq!(disp, 15);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_patch_rel32_jmp_backward() {
        // JMP rel32 at offset 10, target at offset 0.
        // Instruction is 5 bytes, so disp = 0 - 15 = -15.
        let mut code = vec![0u8; 32];
        code[10] = 0xE9; // JMP opcode
        patch_rel32(&mut code, 10, 0).unwrap();
        let disp = i32::from_le_bytes([code[11], code[12], code[13], code[14]]);
        assert_eq!(disp, -15);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_patch_rel32_jcc() {
        // Jcc rel32 at offset 0, target at offset 20.
        // Instruction is 6 bytes (0F 8x + 4-byte disp), so disp = 20 - 6 = 14.
        let mut code = vec![0u8; 32];
        code[0] = 0x0F;
        code[1] = 0x84; // JE rel32
        patch_rel32(&mut code, 0, 20).unwrap();
        let disp = i32::from_le_bytes([code[2], code[3], code[4], code[5]]);
        assert_eq!(disp, 14);
    }

    // -- Architecture-independent tests ----------------------------------------

    #[test]
    fn test_raw_mmap_roundtrip() {
        unsafe {
            let size = sys::page_align(1);
            let ptr = sys::mmap(size, sys::RW).expect("mmap failed");
            *ptr = 42;
            assert_eq!(*ptr, 42);
            sys::munmap(ptr, size);
        }
    }

    #[test]
    fn test_page_align() {
        assert_eq!(sys::page_align(1), sys::PAGE_SIZE);
        assert_eq!(sys::page_align(sys::PAGE_SIZE), sys::PAGE_SIZE);
        assert_eq!(sys::page_align(sys::PAGE_SIZE + 1), sys::PAGE_SIZE * 2);
    }

    #[test]
    fn test_veneer_addr_offset() {
        #[cfg(target_arch = "aarch64")]
        assert_eq!(veneer_addr_offset(), 8);
        #[cfg(target_arch = "x86_64")]
        assert_eq!(veneer_addr_offset(), 6);
    }

    #[test]
    fn test_emit_veneer_stub_size() {
        let mut code = Vec::new();
        emit_veneer_stub(&mut code);
        // Both architectures: 16 bytes (AArch64: 4+4+8, x86-64: 6+8+2)
        assert_eq!(code.len(), 16);
    }

    #[test]
    fn test_profile_hook_mode_default_is_none() {
        assert_eq!(JitConfig::default().profile_hooks, ProfileHookMode::None);
        assert!(
            !JitConfig::default().emit_entry_counters,
            "entry counters must remain opt-in"
        );
    }

    #[test]
    fn test_emit_entry_counters_upgrades_profile_hooks_when_none() {
        let jit = JitCompiler::new(JitConfig {
            emit_entry_counters: true,
            ..JitConfig::default()
        });
        assert_eq!(jit.profile_hooks, ProfileHookMode::CallCounts);
    }

    #[test]
    fn test_explicit_profile_hooks_win_over_emit_entry_counters() {
        let jit = JitCompiler::new(JitConfig {
            profile_hooks: ProfileHookMode::CallCountsAndTiming,
            emit_entry_counters: true,
            ..JitConfig::default()
        });
        assert_eq!(jit.profile_hooks, ProfileHookMode::CallCountsAndTiming);
    }

    #[test]
    fn test_profile_hook_mode_phase2_stubs_classified() {
        // #396 Phase 2: these variants are API-reserved but not
        // implemented in the trampoline emitter. They must all be
        // classified as stubs so compile_raw rejects them with a clear
        // diagnostic rather than silently producing unhooked code.
        //
        // BlockCounts was demoted out of the stub set in #364 Phase 2
        // and BlockCountsAndTiming was demoted in #364 Phase 3 — both
        // now have real trampoline landings exercised by
        // `tests/jit_block_counters.rs` and
        // `tests/jit_block_counters_and_timing.rs`.
        for mode in [
            ProfileHookMode::EdgeCounts,
            ProfileHookMode::BlockFrequency,
            ProfileHookMode::LoopHeads,
        ] {
            assert!(
                profile_hooks_is_phase2_stub(mode),
                "mode {:?} must be classified as a Phase 2 stub",
                mode
            );
            assert!(
                !profile_hooks_enable_counters(mode),
                "mode {:?} must NOT claim to enable function-entry counters \
                 until Phase 2 trampoline lands",
                mode
            );
        }
        // And verify the currently-implemented modes are NOT classified
        // as stubs (so the early-reject does not fire on the happy path).
        for mode in [
            ProfileHookMode::None,
            ProfileHookMode::CallCounts,
            ProfileHookMode::CallCountsAndTiming,
            ProfileHookMode::BlockCounts,
            ProfileHookMode::BlockCountsAndTiming,
        ] {
            assert!(
                !profile_hooks_is_phase2_stub(mode),
                "mode {:?} must NOT be classified as a Phase 2 stub",
                mode
            );
        }
    }

    #[test]
    fn test_profile_hook_mode_block_counts_enables_block_counters() {
        // #364: BlockCounts is the one mode that turns on
        // `profile_hooks_enable_block_counters`. All other modes must
        // remain false here so the BlockCounts-specific splice path is
        // only taken for that exact variant.
        assert!(profile_hooks_enable_block_counters(
            ProfileHookMode::BlockCounts
        ));
        for mode in [
            ProfileHookMode::None,
            ProfileHookMode::CallCounts,
            ProfileHookMode::CallCountsAndTiming,
            ProfileHookMode::BlockCountsAndTiming,
            ProfileHookMode::EdgeCounts,
            ProfileHookMode::BlockFrequency,
            ProfileHookMode::LoopHeads,
        ] {
            assert!(
                !profile_hooks_enable_block_counters(mode),
                "mode {:?} must NOT enable per-block counters",
                mode
            );
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_profile_hook_mode_remaining_stubs_rejected() {
        // The still-stubbed Phase 2 variants must continue to return
        // `ProfileHookModeUnimplemented` — not silently succeed, not
        // return `ProfileHooksUnsupported` (which is specifically the
        // wrong-architecture error).
        //
        // BlockCountsAndTiming moved out of this list in #364 Phase 3
        // once its timing-aware trampoline landed; see
        // `tests/jit_block_counters_and_timing.rs` for happy-path
        // coverage.
        for mode in [
            ProfileHookMode::EdgeCounts,
            ProfileHookMode::BlockFrequency,
            ProfileHookMode::LoopHeads,
        ] {
            let jit = JitCompiler::new(JitConfig {
                profile_hooks: mode,
                ..JitConfig::default()
            });
            let ext: HashMap<String, *const u8> = HashMap::new();
            match jit.compile_raw(&[], &ext) {
                Err(JitError::ProfileHookModeUnimplemented { mode: got }) => {
                    assert_eq!(got, mode);
                }
                Err(other) => panic!(
                    "expected ProfileHookModeUnimplemented for {:?}, got error {:?}",
                    mode, other
                ),
                Ok(_) => panic!(
                    "expected ProfileHookModeUnimplemented for {:?}, got Ok(ExecutableBuffer)",
                    mode
                ),
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_profile_trampoline_byte_layout() {
        let mut code = Vec::new();
        let literal_slot_offset = emit_profile_trampoline_aarch64(&mut code);

        assert_eq!(literal_slot_offset, 20);
        assert_eq!(code.len(), 28);
        assert_eq!(
            u32::from_le_bytes([code[0], code[1], code[2], code[3]]),
            0x5800_00B0
        );
        assert_eq!(
            u32::from_le_bytes([code[4], code[5], code[6], code[7]]),
            0xF940_0211
        );
        assert_eq!(
            u32::from_le_bytes([code[8], code[9], code[10], code[11]]),
            0x9100_0631
        );
        assert_eq!(
            u32::from_le_bytes([code[12], code[13], code[14], code[15]]),
            0xF900_0211
        );
        assert_eq!(
            u32::from_le_bytes([code[16], code[17], code[18], code[19]]),
            0x1400_0003
        );
        assert_eq!(&code[20..28], &[0u8; 8]);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_profile_trampoline_x86_64_byte_layout() {
        // x86-64 entry-counter trampoline (see emit_profile_trampoline_x86_64).
        //
        //   offset 0  : 50                     push rax
        //   offset 1..=2  : 48 B8              movabs rax, imm64 opcode
        //   offset 3..=10 : imm64 placeholder  (patched to heap counter addr)
        //   offset 11..=14: F0 48 FF 00        lock incq qword ptr [rax]
        //   offset 15 : 58                     pop rax
        let mut code = Vec::new();
        let imm64_offset = emit_profile_trampoline_x86_64(&mut code);

        assert_eq!(imm64_offset, 3, "imm64 slot follows push rax + REX/mov");
        assert_eq!(code.len(), 16, "trampoline must be exactly 16 bytes");
        assert_eq!(code[0], 0x50, "push rax");
        assert_eq!(&code[1..3], &[0x48, 0xB8], "movabs rax, imm64 opcode");
        assert_eq!(&code[3..11], &[0u8; 8], "imm64 starts zeroed until patch");
        assert_eq!(
            &code[11..15],
            &[0xF0, 0x48, 0xFF, 0x00],
            "lock incq qword ptr [rax]"
        );
        assert_eq!(code[15], 0x58, "pop rax");
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_profile_hooks_none_has_no_trampoline() {
        let func = build_return_const_named("f");
        let (bytes, fixups) = encode_function_with_fixups(&func).expect("encode should succeed");
        assert!(
            fixups.is_empty(),
            "return-const fixture should encode without fixups"
        );

        let jit = JitCompiler::new(JitConfig::default());
        let ext: HashMap<String, *const u8> = HashMap::new();
        let buf = jit
            .compile_raw(&[func], &ext)
            .expect("compile_raw should succeed");

        let ptr = buf
            .get_fn_ptr_bound("f")
            .expect("function pointer should exist")
            .as_ptr();
        let actual = unsafe { std::slice::from_raw_parts(ptr, bytes.len()) };
        assert_eq!(actual, bytes.as_slice());
        assert!(buf.get_profile("f").is_none());
        assert_eq!(buf.profiles().count(), 0);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_jit_call_counter_increments() {
        let jit = JitCompiler::new(JitConfig {
            profile_hooks: ProfileHookMode::CallCounts,
            ..JitConfig::default()
        });
        let ext: HashMap<String, *const u8> = HashMap::new();
        let buf = jit
            .compile_raw(&[build_return_const_named("f")], &ext)
            .expect("compile_raw should succeed");

        let f: extern "C" fn() -> u64 = unsafe {
            buf.get_fn_bound("f")
                .expect("typed function pointer should exist")
                .into_inner()
        };
        for _ in 0..100 {
            assert_eq!(f(), 42);
        }

        let stats = buf.get_profile("f").expect("profile should exist");
        assert_eq!(stats.call_count, 100);

        let collected: Vec<(&str, ProfileStats)> = buf.profiles().collect();
        assert_eq!(collected.len(), 1);
        assert_eq!(collected[0].0, "f");
        assert_eq!(collected[0].1.call_count, 100);
    }

    #[cfg(unix)]
    #[test]
    fn test_lookup_process_symbol_libc_malloc() {
        // malloc is in libSystem on macOS, libc.so.6 on Linux — always resolvable
        // in any Rust binary since the allocator links libc.
        let ptr = lookup_process_symbol("malloc");
        assert!(
            ptr.is_some(),
            "malloc should be resolvable via dlsym(RTLD_DEFAULT)"
        );
        assert!(!ptr.unwrap().is_null());
    }

    #[cfg(unix)]
    #[test]
    fn test_lookup_process_symbol_missing_returns_none() {
        let ptr = lookup_process_symbol("definitely_not_a_real_symbol_qwertyuiop_12345");
        assert!(ptr.is_none());
    }

    #[cfg(unix)]
    #[test]
    fn test_lookup_process_symbol_interior_nul_returns_none() {
        // CString::new rejects interior NULs.
        let ptr = lookup_process_symbol("mal\0loc");
        assert!(ptr.is_none());
    }

    #[cfg(unix)]
    #[test]
    fn test_extern_symbols_preferred_over_dlsym() {
        // If caller supplies an explicit pointer for "malloc", compile_raw's
        // resolution must use the supplied pointer rather than dlsym'd one.
        // We test the helper directly since building a full compile_raw test
        // here would be heavy; use `resolve_extern` helper.
        let mut map: HashMap<String, *const u8> = HashMap::new();
        let fake_ptr: *const u8 = 0xdead_beef_usize as *const u8;
        map.insert("malloc".to_string(), fake_ptr);
        let resolved = resolve_extern("malloc", &map);
        assert_eq!(
            resolved,
            Some(fake_ptr),
            "extern_symbols must override dlsym"
        );
    }

    #[test]
    fn test_mprotect_rw_then_rx() {
        unsafe {
            let size = sys::page_align(64);
            let ptr = sys::mmap(size, sys::RW).expect("mmap failed");
            // Write a pattern
            for i in 0..64 {
                *ptr.add(i) = i as u8;
            }
            // Switch to RX
            sys::mprotect(ptr, size, sys::RX).expect("mprotect failed");
            // Verify data is still readable
            assert_eq!(*ptr, 0);
            assert_eq!(*ptr.add(63), 63);
            sys::munmap(ptr, size);
        }
    }

    // -- Issue #353: veneer patching with missing extern symbols ----------------

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_patch_branch26_oob_offset_returns_error() {
        let mut code = vec![0u8; 4];
        // offset 4 means code[4..8] which is out of bounds for a 4-byte buffer
        let result = patch_branch26(&mut code, 4, 0);
        assert!(matches!(
            result,
            Err(JitError::FixupOutOfBounds {
                offset: 4,
                code_len: 4
            })
        ));
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_patch_rel32_oob_offset_returns_error() {
        let mut code = vec![0xE8, 0, 0, 0, 0]; // 5-byte CALL
        // offset 2 means we need code[2..8] which is out of bounds
        let result = patch_rel32(&mut code, 2, 100);
        assert!(matches!(
            result,
            Err(JitError::FixupOutOfBounds {
                offset: 2,
                code_len: 5
            })
        ));
    }

    // -- Issue #352: get_fn transmute_copy size check --------------------------

    #[test]
    fn test_get_fn_size_mismatch_panics() {
        // Create a minimal ExecutableBuffer with a known symbol
        let size = sys::page_align(16);
        let memory = unsafe { sys::mmap(size, sys::RW).expect("mmap failed") };
        let mut symbol_offsets = HashMap::new();
        symbol_offsets.insert("test".to_string(), 0u64);
        let buf = ExecutableBuffer {
            memory,
            len: size,
            symbol_offsets,
            canonical_symbols: vec!["test".to_string()],
            counters: HashMap::new(),
            timing_cells: HashMap::new(),
            timing_state: None,
            certificates: HashMap::new(),
        };
        // [u8; 16] is 16 bytes, not pointer-sized (8 bytes) — should panic
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
            buf.get_fn::<[u8; 16]>("test")
        }));
        assert!(result.is_err(), "get_fn should panic on size mismatch");
        // Prevent Drop from double-freeing since we need to clean up manually
        std::mem::forget(buf);
        unsafe {
            sys::munmap(memory, size);
        }
    }

    #[test]
    fn test_get_fn_pointer_sized_ok() {
        let size = sys::page_align(16);
        let memory = unsafe { sys::mmap(size, sys::RW).expect("mmap failed") };
        let mut symbol_offsets = HashMap::new();
        symbol_offsets.insert("test".to_string(), 0u64);
        let buf = ExecutableBuffer {
            memory,
            len: size,
            symbol_offsets,
            canonical_symbols: vec!["test".to_string()],
            counters: HashMap::new(),
            timing_cells: HashMap::new(),
            timing_state: None,
            certificates: HashMap::new(),
        };
        // fn() is pointer-sized — should not panic
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
            buf.get_fn::<fn()>("test")
        }));
        assert!(
            result.is_ok(),
            "get_fn should succeed for pointer-sized types"
        );
        std::mem::forget(buf);
        unsafe {
            sys::munmap(memory, size);
        }
    }

    // -- Issue #345: BL range validation for veneer trampolines ----------------

    /// `branch26_in_range` accepts distances strictly inside +-128 MiB and
    /// rejects anything at or beyond the asymmetric limit [-2^27, +2^27).
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_branch26_in_range_boundaries() {
        // Exactly +128 MiB is the first out-of-range value (imm26 is a signed
        // 26-bit word offset => max byte distance is (1 << 27) - 4).
        let (ok, _) = branch26_in_range(0, AARCH64_BRANCH26_MAX as u64);
        assert!(!ok, "distance of +128 MiB must be out of range");

        // One instruction under the limit is fine.
        let (ok, _) = branch26_in_range(0, (AARCH64_BRANCH26_MAX - 4) as u64);
        assert!(ok, "distance of +128 MiB - 4 must be in range");

        // Zero is trivially in range.
        let (ok, dist) = branch26_in_range(100, 100);
        assert!(ok);
        assert_eq!(dist, 0);
    }

    /// `patch_branch26` must refuse to encode an out-of-range BL instead of
    /// silently truncating the imm26 field.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_patch_branch26_out_of_range_returns_error() {
        // Build a 4-byte BL so the fixup offset itself is in bounds; the
        // target is what overflows +-128 MiB. We cannot actually allocate
        // 128 MiB of code in a test, but `patch_branch26` only inspects the
        // arithmetic distance — any (offset, target) pair whose delta
        // exceeds the limit exercises the error path.
        let mut code = vec![0u8; 4];
        code[0..4].copy_from_slice(&0x9400_0000u32.to_le_bytes());
        let target = AARCH64_BRANCH26_MAX as u64; // exactly +128 MiB, first invalid
        let result = patch_branch26(&mut code, 0, target);
        match result {
            Err(JitError::BranchOutOfRange {
                offset,
                target: t,
                distance,
            }) => {
                assert_eq!(offset, 0);
                assert_eq!(t, target);
                assert_eq!(distance, AARCH64_BRANCH26_MAX);
            }
            other => panic!("expected BranchOutOfRange, got {:?}", other),
        }
        // The instruction must be unchanged — no partial/corrupt encoding.
        let still = u32::from_le_bytes([code[0], code[1], code[2], code[3]]);
        assert_eq!(still, 0x9400_0000);
    }

    /// Construct a `VeneerOutOfRange` error and confirm the diagnostic
    /// message names the symbol, distance, and 128 MiB limit so operators
    /// can tell veneer-distance problems apart from generic branch range
    /// failures (e.g., a cross-function BL that was never reachable).
    #[test]
    fn test_veneer_out_of_range_error_message() {
        let err = JitError::VeneerOutOfRange {
            symbol: "_host_helper".to_string(),
            offset: 0,
            veneer_offset: 256 * 1024 * 1024,
            distance: 256 * 1024 * 1024,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("_host_helper"), "message: {msg}");
        assert!(msg.contains("128MiB"), "message: {msg}");
    }

    /// Drive `validate_veneer_ranges` — the production seam used by
    /// `compile_raw` — with a synthetic `(fixup, veneer)` pair whose distance
    /// is just past the AArch64 imm26 limit. This is the end-to-end regression
    /// for #345: it proves the compile_raw validation path actually returns
    /// `VeneerOutOfRange` rather than falling through to `patch_branch26` and
    /// emitting a silently-truncated BL.
    ///
    /// We cannot allocate >128 MiB of real code in a unit test, so we pass
    /// hand-built offsets directly. On non-aarch64 hosts the validator is a
    /// no-op by design — the imm26 range is an AArch64-specific constraint —
    /// so the test is gated to aarch64.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_validate_veneer_ranges_detects_out_of_range_patch() {
        // BL at offset 0; veneer pretends to live at exactly +128 MiB.
        // +128 MiB is the first out-of-range byte distance (imm26 encodes
        // a signed 26-bit word offset, so the reachable byte range is
        // `[-2^27, +2^27)`).
        let ext_patches = vec![(
            0u32,
            AARCH64_BRANCH26_MAX as u64,
            "_host_helper".to_string(),
        )];
        // `code_len` is only advisory for future island-aware variants;
        // the current validator looks only at the (fixup, veneer) arithmetic.
        let err = validate_veneer_ranges(&ext_patches, AARCH64_BRANCH26_MAX as usize + 16)
            .expect_err("validator must reject out-of-range veneer");
        match err {
            JitError::VeneerOutOfRange {
                symbol,
                offset,
                veneer_offset,
                distance,
            } => {
                assert_eq!(symbol, "_host_helper");
                assert_eq!(offset, 0);
                assert_eq!(veneer_offset, AARCH64_BRANCH26_MAX as u64);
                assert_eq!(distance, AARCH64_BRANCH26_MAX);
            }
            other => panic!("expected VeneerOutOfRange, got {:?}", other),
        }
    }

    /// A veneer exactly 4 bytes inside the +-128 MiB limit must be accepted —
    /// this locks in the asymmetric `[-2^27, +2^27)` boundary and protects
    /// against future off-by-one regressions (e.g., someone flipping `>=` to
    /// `>` or rewriting the limit as `1 << 27 - 1`).
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_validate_veneer_ranges_accepts_boundary_minus_four() {
        let ext_patches = vec![(
            0u32,
            (AARCH64_BRANCH26_MAX - 4) as u64,
            "_host_helper".to_string(),
        )];
        validate_veneer_ranges(&ext_patches, (AARCH64_BRANCH26_MAX + 16) as usize)
            .expect("exact-limit-minus-one-instruction must be in range");
    }

    /// When multiple veneer patches are in play, the validator must fail on
    /// the *first* out-of-range entry and name that specific symbol. This
    /// guarantees the error message points at the actual distance problem
    /// rather than whichever fixup happened to be last in the list.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_validate_veneer_ranges_reports_first_failing_symbol() {
        let ext_patches = vec![
            (0u32, 16u64, "_near".to_string()), // trivially in range
            (0u32, AARCH64_BRANCH26_MAX as u64, "_far".to_string()), // the culprit
            (
                0u32,
                (AARCH64_BRANCH26_MAX * 2) as u64,
                "_further".to_string(),
            ), // also bad, but must not be reported first
        ];
        let err = validate_veneer_ranges(&ext_patches, 0)
            .expect_err("validator must reject when any veneer is out of range");
        match err {
            JitError::VeneerOutOfRange { symbol, .. } => {
                assert_eq!(
                    symbol, "_far",
                    "validator must report the first failing symbol"
                );
            }
            other => panic!("expected VeneerOutOfRange, got {:?}", other),
        }
    }

    // -- Issue #360: symbol_count / symbols robustness -------------------------

    /// Build a minimal `ExecutableBuffer` whose `symbol_offsets` map mirrors
    /// what `compile_raw` produces: one canonical entry per function plus one
    /// Mach-O-style `_name` alias pointing at the same offset. Tests don't
    /// want to run the full codegen pipeline, so we fabricate the map
    /// directly and verify the counting / iteration contracts.
    fn make_buf_with_symbols(names: &[&str]) -> ExecutableBuffer {
        let size = sys::page_align(16);
        let memory = unsafe { sys::mmap(size, sys::RW).expect("mmap failed") };
        let mut symbol_offsets: HashMap<String, u64> = HashMap::new();
        let mut canonical_symbols = Vec::with_capacity(names.len());
        for (i, n) in names.iter().enumerate() {
            let off = i as u64 * 4;
            canonical_symbols.push((*n).to_string());
            symbol_offsets.insert((*n).to_string(), off);
            symbol_offsets.insert(format!("_{}", n), off);
        }
        ExecutableBuffer {
            memory,
            len: size,
            symbol_offsets,
            canonical_symbols,
            counters: HashMap::new(),
            timing_cells: HashMap::new(),
            timing_state: None,
            certificates: HashMap::new(),
        }
    }

    /// Plain names without any underscore prefix must count and enumerate
    /// correctly. This is the common "user function" case.
    #[test]
    fn test_symbol_count_plain_names() {
        let buf = make_buf_with_symbols(&["foo", "bar", "baz"]);
        assert_eq!(buf.symbol_count(), 3);
        let names: Vec<&str> = buf.symbols().map(|(n, _)| n).collect();
        assert!(names.contains(&"foo"));
        assert!(names.contains(&"bar"));
        assert!(names.contains(&"baz"));
        assert_eq!(names.len(), 3);
    }

    /// Mixed plain and `_`-prefixed names: `symbol_count` must still equal
    /// the number of functions, and `symbols()` must yield every canonical
    /// name (including the `_foo` one) exactly once. The old `/ 2` hack
    /// happened to get the count right here but the `starts_with('_')`
    /// filter hid `_priv`.
    #[test]
    fn test_symbol_count_mixed_underscore_prefix() {
        let buf = make_buf_with_symbols(&["foo", "_priv", "bar"]);
        assert_eq!(
            buf.symbol_count(),
            3,
            "symbol_count must not depend on `_name` alias parity"
        );
        let names: Vec<&str> = buf.symbols().map(|(n, _)| n).collect();
        assert!(names.contains(&"foo"));
        assert!(
            names.contains(&"_priv"),
            "symbols() must expose canonical names starting with '_', \
             not silently filter them out"
        );
        assert!(names.contains(&"bar"));
        assert_eq!(names.len(), 3);
    }

    /// All user names start with `_`. Before the fix this case produced a
    /// correct count by coincidence but `symbols()` returned an empty
    /// iterator, hiding every function from callers.
    #[test]
    fn test_symbol_count_all_underscore_prefix() {
        let buf = make_buf_with_symbols(&["_foo", "_bar"]);
        assert_eq!(buf.symbol_count(), 2);
        let names: Vec<&str> = buf.symbols().map(|(n, _)| n).collect();
        assert_eq!(names.len(), 2, "all-underscore names must not be hidden");
        assert!(names.contains(&"_foo"));
        assert!(names.contains(&"_bar"));
    }

    /// `symbols()` must return each function exactly once, even though the
    /// underlying `symbol_offsets` map holds two entries per function
    /// (canonical + `_`-prefixed alias). This is the invariant the old
    /// `/ 2` hack tried — and sometimes failed — to express.
    #[test]
    fn test_symbols_deduplicates_aliases() {
        let buf = make_buf_with_symbols(&["foo", "bar"]);
        let collected: Vec<(&str, u64)> = buf.symbols().collect();
        assert_eq!(collected.len(), 2);
        // Each canonical name should appear once and map to the same offset
        // as its alias in `symbol_offsets`.
        for (name, off) in collected {
            assert_eq!(buf.symbol_offsets.get(name).copied(), Some(off));
            assert_eq!(
                buf.symbol_offsets.get(&format!("_{}", name)).copied(),
                Some(off),
                "alias `_{name}` must resolve to the same offset"
            );
        }
    }

    /// Empty buffer edge case: both accessors should be safe and empty.
    #[test]
    fn test_symbol_count_empty() {
        let buf = make_buf_with_symbols(&[]);
        assert_eq!(buf.symbol_count(), 0);
        assert_eq!(buf.symbols().count(), 0);
    }

    // -- JitConfig dispatch-verification defaults (#375) -----------------------

    /// Tests for the #375 change: `JitConfig::default().verify_dispatch` is
    /// `DispatchVerifyMode::ErrorOnFailure`, the field is caller-visible, and
    /// the value propagates into the underlying `Pipeline`. Uses the
    /// `pub(crate) #[cfg(test)] pipeline()` accessor so tests can reach
    /// `Pipeline::generate_and_verify_dispatch` without exposing a permanent
    /// public API surface.
    mod dispatch_verify_defaults {
        use super::*;
        use crate::pipeline::PipelineError;
        use llvm2_ir::cost_model::CostModelGen;
        use llvm2_lower::compute_graph::{
            ComputeCost, ComputeGraph, ComputeNode, ComputeNodeId, DataEdge, NodeKind,
            TargetRecommendation,
        };
        use llvm2_lower::target_analysis::ComputeTarget;
        use std::collections::HashMap;

        fn make_graph(nodes: Vec<ComputeNode>, edges: Vec<DataEdge>) -> ComputeGraph {
            let mut graph = ComputeGraph::new_with_profitability(CostModelGen::M1);
            graph.nodes = nodes;
            graph.edges = edges;
            graph
        }

        /// A deliberately broken graph: the single node only permits GPU, so
        /// the dispatch verifier cannot construct a safe CPU fallback. This
        /// mirrors `bad_fallback_graph` in `pipeline.rs` tests.
        fn bad_fallback_graph() -> (ComputeGraph, Vec<TargetRecommendation>) {
            let mut gpu_costs = HashMap::new();
            gpu_costs.insert(
                ComputeTarget::Gpu,
                ComputeCost {
                    latency_cycles: 5,
                    throughput_ops_per_kcycle: 100_000,
                },
            );

            let node = ComputeNode {
                id: ComputeNodeId(0),
                instructions: vec![],
                costs: gpu_costs,
                legal_targets: vec![ComputeTarget::Gpu], // No CPU fallback.
                kind: NodeKind::DataParallel,
                data_size_bytes: 4096,
                produced_values: vec![],
                consumed_values: vec![],
                dominant_op: "ADD".to_string(),
                target_legality: None,
                matmul_shape: None,
            };

            let graph = make_graph(vec![node], vec![]);

            let recs = vec![TargetRecommendation {
                node_id: ComputeNodeId(0),
                recommended_target: ComputeTarget::Gpu,
                legal_targets: vec![ComputeTarget::Gpu],
                reason: "GPU only".to_string(),
                parallel_reduction_legal: false,
            }];

            (graph, recs)
        }

        /// AC 2: `JitConfig::default()` selects `ErrorOnFailure`.
        #[test]
        fn default_verify_dispatch_is_error_on_failure() {
            let cfg = JitConfig::default();
            assert_eq!(
                cfg.verify_dispatch,
                DispatchVerifyMode::ErrorOnFailure,
                "JitConfig default must be ErrorOnFailure (#375) so \
                 dispatch-verification failures surface as errors."
            );
        }

        /// AC 1: The `verify_dispatch` field is public and settable.
        #[test]
        fn verify_dispatch_field_is_settable() {
            let cfg = JitConfig {
                verify_dispatch: DispatchVerifyMode::FallbackOnFailure,
                ..JitConfig::default()
            };
            assert_eq!(cfg.verify_dispatch, DispatchVerifyMode::FallbackOnFailure);

            let cfg_off = JitConfig {
                verify_dispatch: DispatchVerifyMode::Off,
                ..JitConfig::default()
            };
            assert_eq!(cfg_off.verify_dispatch, DispatchVerifyMode::Off);
        }

        /// AC 3: Default `JitCompiler` propagates `ErrorOnFailure` into the
        /// underlying pipeline, so a dispatch-failing graph yields
        /// `PipelineError::DispatchVerificationFailed` rather than a silent
        /// CPU-only fallback. Uses the `#[cfg(test)] pipeline()` accessor.
        #[test]
        fn default_jit_propagates_error_mode_into_pipeline() {
            let jit = JitCompiler::new(JitConfig::default());
            let (graph, recs) = bad_fallback_graph();

            let result = jit.pipeline().generate_and_verify_dispatch(&graph, &recs);

            match result {
                Ok(_) => panic!(
                    "Default JitConfig must surface dispatch-verification \
                     failures as errors (#375); bad_fallback_graph should \
                     not pass."
                ),
                Err(PipelineError::DispatchVerificationFailed {
                    violations,
                    summary,
                    report,
                }) => {
                    assert!(violations > 0, "expected at least one violation");
                    assert!(!summary.is_empty(), "expected non-empty summary");
                    assert!(
                        !report.cpu_fallback_ok,
                        "expected the report to flag the missing CPU fallback"
                    );
                }
                Err(other) => {
                    panic!("expected DispatchVerificationFailed, got {other:?}")
                }
            }
        }

        /// Opt-in `FallbackOnFailure` preserves legacy silent-fallback
        /// behaviour end-to-end through the JIT-owned pipeline.
        #[test]
        fn opt_in_fallback_mode_preserves_silent_fallback() {
            let jit = JitCompiler::new(JitConfig {
                verify_dispatch: DispatchVerifyMode::FallbackOnFailure,
                ..JitConfig::default()
            });
            let (graph, recs) = bad_fallback_graph();

            let result = jit.pipeline().generate_and_verify_dispatch(&graph, &recs);
            assert!(
                result.is_ok(),
                "FallbackOnFailure must not surface an error, got {:?}",
                result.err()
            );
        }

        /// `Off` mode bypasses the verifier entirely.
        #[test]
        fn off_mode_skips_verification() {
            let jit = JitCompiler::new(JitConfig {
                verify_dispatch: DispatchVerifyMode::Off,
                ..JitConfig::default()
            });
            let (graph, recs) = bad_fallback_graph();

            let result = jit.pipeline().generate_and_verify_dispatch(&graph, &recs);
            assert!(
                result.is_ok(),
                "Off must not surface an error, got {:?}",
                result.err()
            );
        }
    }
}
