// llvm2-fuzz/src/bin/csmith_driver.rs - csmith differential driver
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Wraps external `csmith` to generate a random C program per iteration,
// compiles it with clang -O0 -S -emit-llvm (oracle producing LLVM IR),
// and pipes the `.ll` through `llvm2-ws2-import` into a Mach-O object.
// Classification per iteration:
//   * importer exit 0                       => ok (real compile to object)
//   * stderr starts with `unsupported:`     => unsupported (importer gap)
//   * stderr starts with `parse:`           => crash (parser bug; saved)
//   * other non-zero exit (panic/signal)    => crash (saved as repro)
//   * clang failure                         => crash (clang_fail; saved)
// The existing "unavailable" status is preserved only for missing tools
// (csmith, clang, or llvm2-ws2-import). Once those exist we actually run.
//
// CLI:
//   csmith-driver --duration SECS --out DIR [--seed-start N]
//                 [--llvm2-ws2-import PATH] [--clang PATH]

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use llvm2_fuzz::runlog::{Repro, RunLog};

const MAX_REPROS_RECORDED: usize = 32;
const PROGRESS_INTERVAL_SECS: u64 = 15;
const PER_ITER_TIMEOUT_SECS: u64 = 60;

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
            return w[1].parse().expect("parse arg");
        }
    }
    default
}

fn iso_now() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let (year, month, day, hh, mm, ss) = decompose_unix(secs);
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        year, month, day, hh, mm, ss
    )
}

fn decompose_unix(secs: u64) -> (u32, u32, u32, u32, u32, u32) {
    let days = (secs / 86_400) as i64;
    let ss = (secs % 60) as u32;
    let mm = ((secs / 60) % 60) as u32;
    let hh = ((secs / 3_600) % 24) as u32;

    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u32;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = (yoe as i64) + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = y + if m <= 2 { 1 } else { 0 };
    (y as u32, m, d, hh, mm, ss)
}

fn tool_exists(name: &str) -> bool {
    // Probe from the tempdir because csmith unconditionally writes a
    // `platform.info` sidecar to cwd on every invocation — even for
    // `--help`/`--version`. Running from env::temp_dir() keeps the
    // repo root clean.
    Command::new(name)
        .arg("--help")
        .current_dir(env::temp_dir())
        .output()
        .map(|o| o.status.success() || !o.stdout.is_empty() || !o.stderr.is_empty())
        .unwrap_or(false)
}

fn command_works(name: &str, arg: &str, require_success: bool) -> bool {
    Command::new(name)
        .arg(arg)
        .output()
        .map(|o| {
            if require_success {
                o.status.success()
            } else {
                o.status.success() || !o.stdout.is_empty() || !o.stderr.is_empty()
            }
        })
        .unwrap_or(false)
}

fn resolve_importer(candidate: &str) -> Option<String> {
    if Path::new(candidate).exists() {
        return Some(candidate.to_string());
    }
    if command_works(candidate, "--help", false) {
        return Some(candidate.to_string());
    }
    if let Some(file_name) = Path::new(candidate).file_name().and_then(|s| s.to_str()) {
        if command_works(file_name, "--help", false) {
            return Some(file_name.to_string());
        }
    }
    None
}

fn resolve_clang(candidate: &str) -> Option<String> {
    if command_works(candidate, "--version", true) {
        return Some(candidate.to_string());
    }
    if let Some(file_name) = Path::new(candidate).file_name().and_then(|s| s.to_str()) {
        if file_name != candidate && command_works(file_name, "--version", true) {
            return Some(file_name.to_string());
        }
    }
    None
}

fn first_line(s: &str) -> String {
    s.lines()
        .next()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .unwrap_or("no stderr")
        .to_string()
}

fn save_repro_source(
    out_dir: &Path,
    driver: &str,
    seed: u64,
    source: &Path,
) -> Option<String> {
    if !source.exists() {
        return None;
    }
    let repro_path = out_dir.join(format!("repro-{}-seed-{}.c", driver, seed));
    fs::copy(source, &repro_path).ok()?;
    Some(repro_path.to_string_lossy().into_owned())
}

fn push_repro(
    log: &mut RunLog,
    out_dir: &Path,
    driver: &str,
    seed: u64,
    summary: String,
    source: Option<&Path>,
) {
    if log.repros.len() >= MAX_REPROS_RECORDED {
        return;
    }
    let minimized_input_path = source.and_then(|path| save_repro_source(out_dir, driver, seed, path));
    log.repros.push(Repro {
        seed,
        minimized_input_path,
        summary,
    });
}

fn write_log(out_dir: &Path, file_name: &str, log: &RunLog) {
    let json_path = out_dir.join(file_name);
    let json = serde_json::to_string_pretty(log).expect("serialize");
    fs::write(&json_path, json).expect("write runlog json");
}

fn find_csmith_include() -> Option<PathBuf> {
    for candidate in [
        "/opt/homebrew/include/csmith-2.3.0",
        "/opt/homebrew/include/csmith",
        "/usr/local/include/csmith-2.3.0",
        "/usr/local/include/csmith",
    ] {
        let path = PathBuf::from(candidate);
        if path.is_dir() {
            return Some(path);
        }
    }
    None
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let duration_secs: u64 = parse_arg(&args, "--duration", 300);
    let out_dir: String = parse_arg(&args, "--out", "evals/results/fuzz/unknown".to_string());
    let seed_start: u64 = parse_arg(&args, "--seed-start", 1);
    let importer_arg: String = parse_arg(
        &args,
        "--llvm2-ws2-import",
        "./target/release/llvm2-ws2-import".to_string(),
    );
    let clang_arg: String = parse_arg(&args, "--clang", "clang".to_string());

    let out_path = PathBuf::from(&out_dir);
    fs::create_dir_all(&out_path).expect("create out_dir");

    let started_at = iso_now();
    let deadline = Instant::now() + Duration::from_secs(duration_secs);
    let campaign_started = Instant::now();

    let mut log = RunLog {
        driver: "csmith-driver".to_string(),
        status: "unavailable".to_string(),
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

    if !tool_exists("csmith") {
        log.reason = Some(
            "csmith not on PATH. Install with: `brew install csmith`. \
             After install, also export CSMITH_PATH to the csmith runtime \
             headers (e.g. /opt/homebrew/include/csmith)."
                .to_string(),
        );
        log.finished_at = iso_now();
        write_log(&out_path, "csmith-driver.json", &log);
        eprintln!("[csmith-driver] unavailable: csmith not on PATH");
        return;
    }

    let importer = match resolve_importer(&importer_arg) {
        Some(path) => path,
        None => {
            log.reason = Some(
                "llvm2-ws2-import not found. Build it with: `cargo build --release -p llvm2-llvm-import`."
                    .to_string(),
            );
            log.finished_at = iso_now();
            write_log(&out_path, "csmith-driver.json", &log);
            eprintln!("[csmith-driver] unavailable: llvm2-ws2-import missing");
            return;
        }
    };

    let clang = match resolve_clang(&clang_arg) {
        Some(path) => path,
        None => {
            log.reason = Some(
                "clang --version failed. Install clang and ensure it is on PATH.".to_string(),
            );
            log.finished_at = iso_now();
            write_log(&out_path, "csmith-driver.json", &log);
            eprintln!("[csmith-driver] unavailable: clang missing or broken");
            return;
        }
    };

    let csmith_include = find_csmith_include();
    if csmith_include.is_none() {
        eprintln!(
            "[csmith-driver] warning: csmith headers not found in standard locations; omitting -I"
        );
    }

    let temp_root = env::temp_dir().join(format!("llvm2-fuzz-csmith-{}", std::process::id()));
    let mut seed = seed_start;
    let mut iter = 0_u64;
    let mut ok_count = 0_u64;
    let mut unsupported_count = 0_u64;
    let mut clang_fail_count = 0_u64;
    let mut last_progress = Instant::now();

    while Instant::now() < deadline {
        let this_seed = seed;
        seed = seed.wrapping_add(1);
        iter = iter.wrapping_add(1);

        let iter_started = Instant::now();
        let iter_dir = temp_root.join(format!("iter-{}", iter));
        let prog_c = iter_dir.join("prog.c");
        let prog_ll = iter_dir.join("prog.ll");
        let prog_o = iter_dir.join("prog.o");

        if fs::create_dir_all(&iter_dir).is_err() {
            log.crashes += 1;
            push_repro(
                &mut log,
                &out_path,
                "csmith-driver",
                this_seed,
                format!("tempdir_create_fail seed={}", this_seed),
                None,
            );
            continue;
        }

        // csmith writes a `platform.info` sidecar to its current working
        // directory regardless of `-o`. Pin cwd to the per-iter tempdir
        // so it ends up inside the dir we clean up, not the repo root.
        let csmith_output = Command::new("csmith")
            .current_dir(&iter_dir)
            .arg("-s")
            .arg(this_seed.to_string())
            .arg("-o")
            .arg(&prog_c)
            .output();

        let csmith_output = match csmith_output {
            Ok(output) => output,
            Err(err) => {
                log.crashes += 1;
                push_repro(
                    &mut log,
                    &out_path,
                    "csmith-driver",
                    this_seed,
                    format!("csmith_fail: seed={} {}", this_seed, err),
                    None,
                );
                let _ = fs::remove_dir_all(&iter_dir);
                continue;
            }
        };

        if !csmith_output.status.success() || !prog_c.exists() {
            log.crashes += 1;
            let stderr = String::from_utf8_lossy(&csmith_output.stderr);
            push_repro(
                &mut log,
                &out_path,
                "csmith-driver",
                this_seed,
                format!("csmith_fail: {} seed={}", first_line(&stderr), this_seed),
                Some(&prog_c),
            );
            let _ = fs::remove_dir_all(&iter_dir);
            continue;
        }

        if iter_started.elapsed() > Duration::from_secs(PER_ITER_TIMEOUT_SECS) {
            log.timeouts += 1;
            let _ = fs::remove_dir_all(&iter_dir);
            continue;
        }

        let mut clang_cmd = Command::new(&clang);
        clang_cmd
            .arg("-O0")
            .arg("-S")
            .arg("-emit-llvm")
            .arg("-w")
            .arg("-o")
            .arg(&prog_ll);
        if let Some(include_dir) = &csmith_include {
            clang_cmd.arg("-I").arg(include_dir);
        }
        clang_cmd.arg(&prog_c);

        let clang_output = match clang_cmd.output() {
            Ok(output) => output,
            Err(err) => {
                clang_fail_count += 1;
                log.crashes += 1;
                push_repro(
                    &mut log,
                    &out_path,
                    "csmith-driver",
                    this_seed,
                    format!("clang_fail: seed={} {}", this_seed, err),
                    Some(&prog_c),
                );
                let _ = fs::remove_dir_all(&iter_dir);
                continue;
            }
        };

        if !clang_output.status.success() || !prog_ll.exists() {
            clang_fail_count += 1;
            log.crashes += 1;
            let stderr = String::from_utf8_lossy(&clang_output.stderr);
            push_repro(
                &mut log,
                &out_path,
                "csmith-driver",
                this_seed,
                format!("clang_fail: {} seed={}", first_line(&stderr), this_seed),
                Some(&prog_c),
            );
            let _ = fs::remove_dir_all(&iter_dir);
            continue;
        }

        if iter_started.elapsed() > Duration::from_secs(PER_ITER_TIMEOUT_SECS) {
            log.timeouts += 1;
            let _ = fs::remove_dir_all(&iter_dir);
            continue;
        }

        let import_output = match Command::new(&importer)
            .arg(&prog_ll)
            .arg(&prog_o)
            .output()
        {
            Ok(output) => output,
            Err(err) => {
                log.crashes += 1;
                push_repro(
                    &mut log,
                    &out_path,
                    "csmith-driver",
                    this_seed,
                    format!("import_fail: seed={} {}", this_seed, err),
                    Some(&prog_c),
                );
                let _ = fs::remove_dir_all(&iter_dir);
                continue;
            }
        };

        let stderr = String::from_utf8_lossy(&import_output.stderr);
        let summary_line = first_line(&stderr);

        log.runs += 1;
        if import_output.status.success() {
            ok_count += 1;
        } else if stderr.starts_with("unsupported:") {
            unsupported_count += 1;
        } else if stderr.starts_with("parse:") {
            log.crashes += 1;
            push_repro(
                &mut log,
                &out_path,
                "csmith-driver",
                this_seed,
                format!("{} seed={}", summary_line, this_seed),
                Some(&prog_c),
            );
        } else {
            log.crashes += 1;
            push_repro(
                &mut log,
                &out_path,
                "csmith-driver",
                this_seed,
                format!("{} seed={}", summary_line, this_seed),
                Some(&prog_c),
            );
        }

        if last_progress.elapsed() >= Duration::from_secs(PROGRESS_INTERVAL_SECS) {
            eprintln!(
                "[csmith-driver] runs={} ok={} unsupported={} crashes={} elapsed={}",
                log.runs,
                ok_count,
                unsupported_count,
                log.crashes,
                campaign_started.elapsed().as_secs()
            );
            last_progress = Instant::now();
        }

        let _ = fs::remove_dir_all(&iter_dir);
    }

    let _ = fs::remove_dir_all(&temp_root);

    log.status = if log.runs > 0 {
        "ok".to_string()
    } else {
        "unavailable".to_string()
    };
    log.reason = Some(format!(
        "runs={} ok={} unsupported={} clang_fail={} crashes={}",
        log.runs, ok_count, unsupported_count, clang_fail_count, log.crashes
    ));
    log.finished_at = iso_now();
    write_log(&out_path, "csmith-driver.json", &log);
    eprintln!(
        "[csmith-driver] wrote {} runs={} ok={} unsupported={} clang_fail={} crashes={}",
        out_path.join("csmith-driver.json").display(),
        log.runs,
        ok_count,
        unsupported_count,
        clang_fail_count,
        log.crashes
    );
}
