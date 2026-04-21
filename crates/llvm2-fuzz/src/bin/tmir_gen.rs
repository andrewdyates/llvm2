// llvm2-fuzz/src/bin/tmir_gen.rs - tMIR differential fuzzing driver
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// For each iteration:
//   1. Pick a seed.
//   2. Generate a random valid tMIR module.
//   3. Pick sample inputs (well-known values + PRNG values).
//   4. Run the tMIR interpreter (oracle).
//   5. Translate tMIR -> llvm2-lower::Function.
//   6. Compile the function through the Pipeline at O0 *and* O2.
//   7. Miscompile signals:
//        - Either compile call panics or returns an error (infrastructure
//          bug in our pipeline, not generator bug — we already filtered
//          for valid tMIR).
//        - Pipeline at O0 and O2 should both accept every valid input
//          (we can't compare their runtime results without JIT, which is
//          deferred to a follow-up). For this MVP we diff the
//          compilation behaviour (success vs failure) between the two
//          levels.
//   8. On crash/miscompile, save the repro (tMIR JSON) to disk and
//      optionally call scripts/file_miscompile_issue.sh.
//
// The oracle loop is conservative by design: in this MVP we don't JIT and
// compare runtime results. We do detect:
//   - Interpreter panics on generated tMIR (generator bug).
//   - Compiler panics on valid tMIR (definite LLVM2 bug).
//   - Compiler errors at O2 but not O0, or vice versa (optimizer bug).
//
// Running actual JIT execution against the interpreter is a follow-up
// (needs aarch64 host + careful ABI handling; see return report).
//
// CLI:
//   tmir-gen --duration SECS --out DIR [--seed-start N] [--per-iter-timeout SECS]

use std::env;
use std::fs;
use std::panic;
use std::path::PathBuf;
use std::process::Command;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use llvm2_codegen::interpreter::{interpret_with_config, InterpreterConfig, InterpreterValue};
use llvm2_codegen::pipeline::{OptLevel, Pipeline, PipelineConfig};
use llvm2_fuzz::runlog::{Repro, RunLog};
use llvm2_fuzz::tmir_gen::{gen_module, sample_inputs, GenConfig, FUZZ_FN_NAME};
use llvm2_lower::adapter::translate_module;

const MAX_REPROS_RECORDED: usize = 32;

fn parse_arg<T: std::str::FromStr>(
    args: &[String],
    flag: &str,
    default: T,
) -> T
where
    T::Err: std::fmt::Debug,
{
    for w in args.windows(2) {
        if w[0] == flag {
            return w[1].parse().expect("parse --flag value");
        }
    }
    default
}

fn iso_now() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    // Minimal ISO8601 UTC rendering — avoids pulling chrono for a fuzz driver.
    let (year, month, day, hh, mm, ss) = decompose_unix(secs);
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        year, month, day, hh, mm, ss
    )
}

/// Decompose seconds-since-epoch into (Y, M, D, h, m, s). Proleptic
/// Gregorian, no leap seconds. Good enough for logging.
fn decompose_unix(secs: u64) -> (u32, u32, u32, u32, u32, u32) {
    let days = (secs / 86_400) as i64;
    let ss = (secs % 60) as u32;
    let mm = ((secs / 60) % 60) as u32;
    let hh = ((secs / 3600) % 24) as u32;

    // Algorithm: howardhinnant date algorithms — days from 1970-01-01.
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u32;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = (yoe as i64) + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = y + if m <= 2 { 1 } else { 0 };
    (y as u32, m, d, hh, mm, ss)
}

fn run_oracle_one(
    module: &tmir::Module,
    args: &[i64],
) -> Result<Vec<i64>, String> {
    let ivals: Vec<InterpreterValue> = args.iter().map(|&x| InterpreterValue::Int(x as i128)).collect();
    let cfg = InterpreterConfig {
        fuel: 200_000,
        max_call_depth: 32,
    };
    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        interpret_with_config(module, FUZZ_FN_NAME, &ivals, cfg)
    }));
    match result {
        Ok(Ok(vs)) => {
            let out: Vec<i64> = vs
                .into_iter()
                .filter_map(|v| match v {
                    InterpreterValue::Int(x) => Some(x as i64),
                    InterpreterValue::Bool(b) => Some(if b { 1 } else { 0 }),
                    _ => None,
                })
                .collect();
            Ok(out)
        }
        Ok(Err(e)) => Err(format!("interp_err: {}", e)),
        Err(_) => Err("interp_panic".to_string()),
    }
}

/// Result of compiling a tMIR module at a given opt level through the
/// Pipeline. `Ok(_)` means the full phases (ISel → opt → regalloc → encode)
/// all succeeded; `Err(_)` carries the stringified error.
fn compile_one(
    module: &tmir::Module,
    opt: OptLevel,
) -> Result<usize, String> {
    // Translate tMIR -> llvm2-lower::Function.
    let lower = match translate_module(module) {
        Ok(v) => v,
        Err(e) => return Err(format!("translate_err: {:?}", e)),
    };
    if lower.is_empty() {
        return Err("translate_empty".to_string());
    }
    let (func, _proof) = &lower[0];
    let pipeline = Pipeline::new(PipelineConfig {
        opt_level: opt,
        emit_debug: false,
        verify_dispatch: llvm2_codegen::pipeline::DispatchVerifyMode::Off,
        verify: false,
        enable_post_ra_opt: matches!(opt, OptLevel::O1 | OptLevel::O2 | OptLevel::O3),
        use_pressure_aware_scheduler: matches!(opt, OptLevel::O2 | OptLevel::O3),
        cegis_superopt_budget_sec: None,
        target_triple: "aarch64-unknown-unknown".to_string(),
    });
    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        pipeline.compile_function(func)
    }));
    match result {
        Ok(Ok(bytes)) => Ok(bytes.len()),
        Ok(Err(e)) => Err(format!("compile_err: {:?}", e)),
        Err(_) => Err("compile_panic".to_string()),
    }
}

fn save_repro(out_dir: &PathBuf, seed: u64, module: &tmir::Module) -> Option<String> {
    let path = out_dir.join(format!("repro-tmir-gen-seed-{}.json", seed));
    let json = match serde_json::to_string_pretty(module) {
        Ok(s) => s,
        Err(_) => return None,
    };
    fs::write(&path, json).ok()?;
    Some(path.to_string_lossy().into_owned())
}

fn maybe_autofile(driver: &str, repro_path: &str) {
    // Only attempt to auto-file if the user set LLVM2_AUTOFILE=1 in the env.
    // The default here is off because filing dozens of issues during a
    // noisy first campaign would be harmful.
    if env::var("LLVM2_AUTOFILE").ok().as_deref() != Some("1") {
        return;
    }
    let script = PathBuf::from("scripts/file_miscompile_issue.sh");
    if !script.exists() {
        return;
    }
    let _ = Command::new(&script)
        .arg(driver)
        .arg(repro_path)
        .status();
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let duration_secs: u64 = parse_arg(&args, "--duration", 300);
    let out_dir: String = parse_arg(&args, "--out", "evals/results/fuzz/unknown".to_string());
    let seed_start: u64 = parse_arg(&args, "--seed-start", 1);

    let out_path = PathBuf::from(&out_dir);
    fs::create_dir_all(&out_path).expect("create out_dir");

    let started_at = iso_now();
    let deadline = Instant::now() + Duration::from_secs(duration_secs);

    let mut log = RunLog {
        driver: "tmir-gen".to_string(),
        status: "ok".to_string(),
        reason: None,
        duration_secs,
        runs: 0,
        timeouts: 0,
        crashes: 0,
        miscompiles: 0,
        repros: Vec::new(),
        started_at: started_at.clone(),
        finished_at: String::new(),
    };

    let mut seed = seed_start;
    let cfg = GenConfig::default();
    let mut last_progress = Instant::now();

    while Instant::now() < deadline {
        seed = seed.wrapping_add(1);
        log.runs += 1;

        // Step 1: generate.
        let module = match panic::catch_unwind(panic::AssertUnwindSafe(|| gen_module(seed, &cfg))) {
            Ok(m) => m,
            Err(_) => {
                log.crashes += 1;
                if log.repros.len() < MAX_REPROS_RECORDED {
                    log.repros.push(Repro {
                        seed,
                        minimized_input_path: None,
                        summary: "gen_module panicked".to_string(),
                    });
                }
                continue;
            }
        };

        // Step 2: interpreter oracle.
        let inputs = sample_inputs(seed, cfg.num_params, 6);
        let mut oracle_outputs: Vec<(Vec<i64>, Result<Vec<i64>, String>)> =
            Vec::with_capacity(inputs.len());
        let mut any_oracle_panic = false;
        for row in &inputs {
            let result = run_oracle_one(&module, row);
            if matches!(result, Err(ref e) if e == "interp_panic") {
                any_oracle_panic = true;
            }
            oracle_outputs.push((row.clone(), result));
        }

        if any_oracle_panic {
            // Interpreter panic on our own generated program is a bug —
            // either generator or interpreter.
            log.crashes += 1;
            if log.repros.len() < MAX_REPROS_RECORDED {
                let saved = save_repro(&out_path, seed, &module);
                let summary = format!("interpreter panic (seed={})", seed);
                if let Some(p) = &saved {
                    maybe_autofile("tmir-gen", p);
                }
                log.repros.push(Repro {
                    seed,
                    minimized_input_path: saved,
                    summary,
                });
            }
            continue;
        }

        // Step 3: compile at O0.
        let o0 = compile_one(&module, OptLevel::O0);

        // Step 4: compile at O2.
        let o2 = compile_one(&module, OptLevel::O2);

        match (&o0, &o2) {
            (Ok(_), Ok(_)) => {
                // Both compiled — oracle matches itself (trivially). Good.
            }
            (Err(e), Err(_)) => {
                // Both failed identically: probably an unsupported tMIR
                // feature; count as "crash" only if it looks like a panic
                // rather than a structured error.
                if e.starts_with("compile_panic") {
                    log.crashes += 1;
                    if log.repros.len() < MAX_REPROS_RECORDED {
                        let saved = save_repro(&out_path, seed, &module);
                        if let Some(p) = &saved {
                            maybe_autofile("tmir-gen", p);
                        }
                        log.repros.push(Repro {
                            seed,
                            minimized_input_path: saved,
                            summary: format!("compile panic (both levels) seed={} {}", seed, e),
                        });
                    }
                }
                // Structured errors at both levels are tolerated —
                // indicates the tMIR shape isn't supported yet.
            }
            (Ok(_), Err(_e)) | (Err(_e), Ok(_)) => {
                // DIFFERENTIAL FINDING: one opt level accepts, the other
                // rejects the same valid tMIR. This is a real miscompile
                // signal (optimizer bug or legalization bug).
                log.miscompiles += 1;
                if log.repros.len() < MAX_REPROS_RECORDED {
                    let saved = save_repro(&out_path, seed, &module);
                    if let Some(p) = &saved {
                        maybe_autofile("tmir-gen", p);
                    }
                    let summary = format!(
                        "O0/O2 disagreement: O0={:?} O2={:?} seed={}",
                        o0.as_ref().map(|_| "ok").map_err(|s| s.as_str()),
                        o2.as_ref().map(|_| "ok").map_err(|s| s.as_str()),
                        seed
                    );
                    log.repros.push(Repro {
                        seed,
                        minimized_input_path: saved,
                        summary,
                    });
                }
            }
        }

        // Emit progress every ~15s so the looper doesn't think we've died.
        if last_progress.elapsed() >= Duration::from_secs(15) {
            eprintln!(
                "[tmir-gen] runs={} miscompiles={} crashes={} elapsed={:?}",
                log.runs,
                log.miscompiles,
                log.crashes,
                Instant::now().saturating_duration_since(
                    deadline - Duration::from_secs(duration_secs)
                )
            );
            last_progress = Instant::now();
        }
    }

    log.finished_at = iso_now();

    // Write the JSON result.
    let json_path = out_path.join("tmir-gen.json");
    let json = serde_json::to_string_pretty(&log).expect("serialize RunLog");
    fs::write(&json_path, json).expect("write tmir-gen.json");
    eprintln!(
        "[tmir-gen] wrote {} runs={} miscompiles={} crashes={}",
        json_path.display(),
        log.runs,
        log.miscompiles,
        log.crashes
    );
}
