// rustc_codegen_llvm2 — M0 skeleton.
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc.
// License: Apache-2.0
//
// Status: WS4 milestone M0 — the crate compiles as a nightly rustc
// codegen backend dylib, rustc can load it via `-Zcodegen-backend=…`,
// and the backend answers every trait method without panicking. Real
// codegen still issues a `fatal` diagnostic; that is the correct M0
// behaviour because a silent `unimplemented!()` from inside a
// codegen-backend dylib turns into an ICE message that is noisier than
// a normal compile failure.
//
// See `designs/2026-04-19-proving-llvm2-replaces-llvm.md` for how this
// slots into WS4.
//
// Reference implementation: the in-tree `rustc_codegen_cranelift` crate
// at
// `$(rustup run nightly rustc --print sysroot)/lib/rustlib/rustc-src/rust/
//  compiler/rustc_codegen_cranelift/src/lib.rs`.
// That tree is what this file is shaped against and is where to look
// when `CodegenBackend` drifts under us.
//
// What intentionally still does not work (M1+):
//   * `codegen_crate` raises a fatal diagnostic instead of actually
//     lowering rustc MIR → tMIR → machine code. The MIR-to-tMIR adapter
//     is tracked under the WS4 M1 task list.
//   * `join_codegen` is unreachable by construction: rustc aborts after
//     a fatal diagnostic before ever calling it. We still implement it
//     (returning empty `CompiledModules`) so that the trait shape is
//     correct and so future milestones can fill it in without touching
//     the trait plumbing.
//   * We deliberately rely on the default `link` impl from
//     `CodegenBackend`. It runs `link_binary` on the
//     `CompiledModules` we produced; with an empty module list it does
//     the right thing (no-op) on top of a fatal diagnostic.

#![feature(rustc_private)]
#![warn(rust_2018_idioms)]
#![warn(unreachable_pub)]
#![warn(unused_lifetimes)]

// rustc internal crates we link against. Keep this list minimal at M0:
// pulling in a crate we don't actually use still forces the linker to
// resolve its symbols against the running rustc's sysroot, which is how
// "dylib load failure" bugs show up. Only add a crate here when it is
// actually used below.
extern crate rustc_codegen_ssa;
extern crate rustc_data_structures;
extern crate rustc_middle;
extern crate rustc_session;

// Prevents duplicating functions and statics that are already part of
// the host rustc process. Matches cranelift's pattern.
#[allow(unused_extern_crates)]
extern crate rustc_driver;

use std::any::Any;

use rustc_codegen_ssa::traits::CodegenBackend;
use rustc_codegen_ssa::{CompiledModules, CrateInfo, TargetConfig};
use rustc_data_structures::fx::FxIndexMap;
use rustc_middle::dep_graph::{WorkProduct, WorkProductId};
use rustc_middle::ty::TyCtxt;
use rustc_session::Session;
use rustc_session::config::OutputFilenames;

/// The LLVM2 codegen backend.
///
/// At M0 this is a near-empty shell. Future milestones will accumulate
/// state here (e.g., an `OnceCell<BackendConfig>` mirroring cranelift,
/// a `tmir-from-rustc-mir` adapter, etc.). We keep the struct non-empty
/// (`version_string`) so that future fields can be added without
/// changing the shape of `__rustc_codegen_backend`.
pub struct Llvm2CodegenBackend {
    version_string: &'static str,
}

impl Llvm2CodegenBackend {
    fn new() -> Self {
        Self {
            version_string: concat!(
                "rustc_codegen_llvm2 ",
                env!("CARGO_PKG_VERSION"),
                " (M0 skeleton — codegen unimplemented)"
            ),
        }
    }
}

impl CodegenBackend for Llvm2CodegenBackend {
    fn name(&self) -> &'static str {
        "llvm2"
    }

    fn init(&self, _sess: &Session) {
        // Deliberately empty at M0. `rustc_codegen_cranelift` parses
        // `-Cllvm-args` here; we don't take any flags yet.
    }

    fn target_config(&self, _sess: &Session) -> TargetConfig {
        // Report a conservative, honest target config. We claim no
        // target features and no reliable f16/f128. When M1 wires up
        // real codegen, we will mirror cranelift's arch-dependent
        // logic here.
        TargetConfig {
            target_features: vec![],
            unstable_target_features: vec![],
            has_reliable_f16: false,
            has_reliable_f16_math: false,
            has_reliable_f128: false,
            has_reliable_f128_math: false,
        }
    }

    fn print_version(&self) {
        println!("{}", self.version_string);
    }

    fn target_cpu(&self, sess: &Session) -> String {
        // Mirror cranelift's policy: honour `-Ctarget-cpu=...` when
        // given, otherwise fall back to the per-target default. We do
        // not understand `-Ctarget-cpu=native` yet; that gets handled
        // by whichever arch-specific backend we dispatch to in M2+.
        match sess.opts.cg.target_cpu {
            Some(ref name) => name.clone(),
            None => sess.target.cpu.as_ref().to_owned(),
        }
    }

    fn codegen_crate(&self, tcx: TyCtxt<'_>, _crate_info: &CrateInfo) -> Box<dyn Any> {
        // M0: we have nothing to compile with yet. Emit a fatal
        // diagnostic so rustc exits cleanly with a non-zero status
        // rather than ICE-ing. `dcx().fatal(...)` never returns, so we
        // don't need a dummy return value — but the signature still
        // requires `Box<dyn Any>`, so we annotate with the `!` path.
        //
        // Tracking issue: https://github.com/andrewdyates/LLVM2/issues/438
        tcx.dcx().fatal(
            "rustc_codegen_llvm2: codegen is not implemented yet (M0 skeleton). \
             Track progress in GitHub issue #438.",
        );
    }

    fn join_codegen(
        &self,
        _ongoing_codegen: Box<dyn Any>,
        _sess: &Session,
        _outputs: &OutputFilenames,
    ) -> (CompiledModules, FxIndexMap<WorkProductId, WorkProduct>) {
        // Unreachable by design: `codegen_crate` always aborts with a
        // fatal diagnostic at M0. If rustc ever calls us here, the
        // fatal diagnostic was silently ignored — that would be a bug
        // in rustc or a contract change we need to notice, so panic
        // loudly.
        unreachable!(
            "rustc_codegen_llvm2::join_codegen invoked, but codegen_crate should have \
             issued a fatal diagnostic first. This indicates a contract mismatch with \
             rustc_codegen_ssa — please file an issue."
        )
    }

    // Intentionally no `link` override — we use the default provided by
    // `CodegenBackend`, which calls `link_binary` on the CompiledModules
    // produced by `join_codegen`. At M0 we never reach it.
}

/// The entrypoint rustc looks up in the dylib via `-Zcodegen-backend`.
///
/// Must be `no_mangle` + `extern` ABI and return `Box<dyn CodegenBackend>`.
/// This symbol name is the contract between rustc and every third-party
/// codegen backend.
#[unsafe(no_mangle)]
pub fn __rustc_codegen_backend() -> Box<dyn CodegenBackend> {
    Box::new(Llvm2CodegenBackend::new())
}
