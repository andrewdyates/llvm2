// llvm2-fuzz - Differential fuzzing for LLVM2
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Differential fuzzing infrastructure for LLVM2. Three drivers live in
// `src/bin/*.rs`; shared helpers (PRNG, tMIR generation, JSON run log
// schema) live here.
//
// Part of WS3 (differential fuzzing) in the "proving llvm2 replaces llvm"
// plan.  See issue referenced in commits.

pub mod prng;
pub mod tmir_gen;
pub mod runlog;
