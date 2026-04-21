// llvm2-ws2-import — WS2 driver helper.
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0
//
// Reads a `.ll` file, imports it to `tmir::Module`, and compiles through
// `llvm2-codegen` to a Mach-O object file.
//
// Contract (consumed by `scripts/run_llvm_test_suite.sh`):
//
//   llvm2-ws2-import <input.ll> <output.o>
//
// Exit codes:
//   0 — object written successfully.
//   1 — importer or codegen error. On `Error::Unsupported(reason)` the
//       first stderr line is `unsupported: <reason>` so the driver can
//       classify the program as `unsupported` rather than `crash`.

use std::env;
use std::fs;
use std::path::Path;
use std::process::ExitCode;

use llvm2_codegen::compiler::{Compiler, CompilerConfig};
use llvm2_codegen::pipeline::OptLevel;
use llvm2_codegen::target::Target;
use llvm2_llvm_import::{Error, import_module};

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: {} <input.ll> <output.o>", args[0]);
        return ExitCode::from(1);
    }
    let input = Path::new(&args[1]);
    let output = Path::new(&args[2]);

    let module = match import_module(input) {
        Ok(m) => m,
        Err(Error::Unsupported(reason)) => {
            // Agreed prefix with scripts/run_llvm_test_suite.sh.
            eprintln!("unsupported: {reason}");
            return ExitCode::from(1);
        }
        Err(Error::Parse { line, message }) => {
            eprintln!("parse: line {line}: {message}");
            return ExitCode::from(1);
        }
        Err(Error::Io(e)) => {
            eprintln!("io: {e}");
            return ExitCode::from(1);
        }
    };

    // O0 matches clang -O0 on the reference side. Target is AArch64
    // because that's the fully functional backend on the host (Apple
    // Silicon). x86-64 is not yet production-grade (see design doc).
    let cfg = CompilerConfig {
        opt_level: OptLevel::O0,
        target: Target::Aarch64,
        emit_proofs: false,
        trace_level: llvm2_codegen::compiler::CompilerTraceLevel::None,
        emit_debug: false,
        parallel: false,
        cegis_superopt_budget_sec: None,
    };
    let compiler = Compiler::new(cfg);
    let result = match compiler.compile(&module) {
        Ok(r) => r,
        Err(e) => {
            // The driver catalogues these as `unsupported` too — the
            // importer admitted the program but the codegen pipeline
            // could not finish lowering something. That's a truthful
            // signal ("we don't support this yet"), not a crash.
            eprintln!("unsupported: codegen: {e}");
            return ExitCode::from(1);
        }
    };

    if let Err(e) = fs::write(output, &result.object_code) {
        eprintln!("io: writing {}: {}", output.display(), e);
        return ExitCode::from(1);
    }

    ExitCode::SUCCESS
}
