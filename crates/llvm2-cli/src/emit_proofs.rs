// llvm2-cli/emit_proofs.rs - Per-proof SMT-LIB2 + certificate emission (#421)
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0
//
// Implements the `--emit-proofs=<dir>` CLI flag introduced in issue #421
// (epic #407, task 6). For every verified lowering rule produced by the
// compiler we write two files to `<dir>/<ProofCategory>/<proof_name>`:
//
//   - `.smt2`  : complete SMT-LIB2 query (via `llvm2_verify::serialize_to_smt2`)
//   - `.cert`  : minimal JSON metadata capturing
//                `{ result, solver, timestamp, hash, proof_name, category }`
//
// Downstream consumers: `tla2`, `tRust` (issues #260, #269).
//
// Design notes:
// * Certificates produced by the codegen pipeline (see
//   `llvm2_codegen::compiler::ProofCertificate`) only carry `rule_name`,
//   `verified`, `category` (String), `strength` (String) and `function_name`
//   fields. To produce real SMT-LIB2 text we need the underlying
//   `ProofObligation` — we reconstruct the mapping by loading the full
//   `ProofDatabase` once and looking up obligations by name.
// * Rules that are verified by codegen but absent from the database (should
//   be rare) are still logged via a `.cert` file with `result: "eval-only"`
//   and no `.smt2`, keeping the audit trail complete.
// * All filenames are sanitized so category variants with spaces or slashes
//   (e.g. "Floating-Point") produce filesystem-safe directory names.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use llvm2_codegen::compiler::ProofCertificate as CodegenCertificate;
use llvm2_verify::{CategorizedProof, ProofDatabase, serialize_to_smt2};

/// Summary of how many proof files were written.
#[derive(Debug, Default, Clone, Copy)]
pub struct EmitSummary {
    pub smt2_written: usize,
    pub cert_written: usize,
    pub skipped_no_obligation: usize,
}

impl EmitSummary {
    pub fn merge(&mut self, other: EmitSummary) {
        self.smt2_written += other.smt2_written;
        self.cert_written += other.cert_written;
        self.skipped_no_obligation += other.skipped_no_obligation;
    }
}

/// Sanitize a string for use as a filesystem path segment.
fn sanitize_path(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            c if c.is_ascii_alphanumeric() => out.push(c),
            '_' | '-' | '.' => out.push(c),
            _ => out.push('_'),
        }
    }
    if out.is_empty() {
        out.push_str("unknown");
    }
    out
}

/// Escape a string for JSON string output.
fn escape_json(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out
}

/// Compute a stable 64-bit FNV-1a hash of a byte slice.
///
/// Chosen over `DefaultHasher` for determinism across Rust versions — the
/// hash lands in the `.cert` JSON and is used downstream for cache
/// invalidation (see `tla2` / `tRust` integration).
fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h
}

/// Build a lookup table from proof name to `(ProofObligation, category name)`.
fn build_obligation_index(db: &ProofDatabase) -> HashMap<String, (usize, &'static str)> {
    let mut map = HashMap::with_capacity(db.len());
    for (idx, cp) in db.all().iter().enumerate() {
        let cat_name = category_dir_name(cp);
        map.insert(cp.obligation.name.clone(), (idx, cat_name));
    }
    map
}

/// Return the category's canonical variant name (e.g. "Arithmetic") for
/// use as a directory name. We use `{:?}` so the directory mirrors the
/// Rust enum variant exactly (no spaces, stable across releases).
fn category_dir_name(cp: &CategorizedProof) -> &'static str {
    // `format!("{:?}", cp.category)` would work but allocates — instead we
    // match against `ProofCategory::name()` at leaf level. For simplicity,
    // we leak the debug repr of the variant via a static dispatch table.
    match cp.category {
        llvm2_verify::ProofCategory::Arithmetic => "Arithmetic",
        llvm2_verify::ProofCategory::Division => "Division",
        llvm2_verify::ProofCategory::FloatingPoint => "FloatingPoint",
        llvm2_verify::ProofCategory::NzcvFlags => "NzcvFlags",
        llvm2_verify::ProofCategory::Comparison => "Comparison",
        llvm2_verify::ProofCategory::Branch => "Branch",
        llvm2_verify::ProofCategory::Peephole => "Peephole",
        llvm2_verify::ProofCategory::Optimization => "Optimization",
        llvm2_verify::ProofCategory::ConstantFolding => "ConstantFolding",
        llvm2_verify::ProofCategory::CopyPropagation => "CopyPropagation",
        llvm2_verify::ProofCategory::CseLicm => "CseLicm",
        llvm2_verify::ProofCategory::DeadCodeElimination => "DeadCodeElimination",
        llvm2_verify::ProofCategory::CfgSimplification => "CfgSimplification",
        llvm2_verify::ProofCategory::Memory => "Memory",
        llvm2_verify::ProofCategory::LoadStoreLowering => "LoadStoreLowering",
        llvm2_verify::ProofCategory::SwitchLowering => "SwitchLowering",
        llvm2_verify::ProofCategory::NeonLowering => "NeonLowering",
        llvm2_verify::ProofCategory::NeonEncoding => "NeonEncoding",
        llvm2_verify::ProofCategory::Vectorization => "Vectorization",
        llvm2_verify::ProofCategory::AnePrecision => "AnePrecision",
        llvm2_verify::ProofCategory::RegAlloc => "RegAlloc",
        llvm2_verify::ProofCategory::BitwiseShift => "BitwiseShift",
        llvm2_verify::ProofCategory::ConstantMaterialization => "ConstantMaterialization",
        llvm2_verify::ProofCategory::AddressMode => "AddressMode",
        llvm2_verify::ProofCategory::FrameLayout => "FrameLayout",
        llvm2_verify::ProofCategory::InstructionScheduling => "InstructionScheduling",
        llvm2_verify::ProofCategory::MachOEmission => "MachOEmission",
        llvm2_verify::ProofCategory::LoopOptimization => "LoopOptimization",
        llvm2_verify::ProofCategory::StrengthReduction => "StrengthReduction",
        llvm2_verify::ProofCategory::CmpCombine => "CmpCombine",
        llvm2_verify::ProofCategory::Gvn => "Gvn",
        llvm2_verify::ProofCategory::TailCallOptimization => "TailCallOptimization",
        llvm2_verify::ProofCategory::IfConversion => "IfConversion",
        llvm2_verify::ProofCategory::FpConversion => "FpConversion",
        llvm2_verify::ProofCategory::ExtensionTruncation => "ExtensionTruncation",
        llvm2_verify::ProofCategory::AtomicOperations => "AtomicOperations",
        llvm2_verify::ProofCategory::CallLowering => "CallLowering",
        llvm2_verify::ProofCategory::X8664Lowering => "X8664Lowering",
    }
}

/// Write one `.cert` JSON file for a codegen certificate.
fn write_cert_file(
    path: &Path,
    cert: &CodegenCertificate,
    smt2_hash: Option<u64>,
    result_tag: &str,
) -> std::io::Result<()> {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let solver = if cert.strength.contains("Statistical") {
        "mock_statistical"
    } else if cert.strength.contains("Exhaustive") {
        "mock_exhaustive"
    } else if cert.strength.contains("Formal") {
        "z4"
    } else {
        "unknown"
    };

    let hash_str = match smt2_hash {
        Some(h) => format!("{}", h),
        None => "null".to_string(),
    };

    let body = format!(
        "{{\n  \"result\": \"{}\",\n  \"solver\": \"{}\",\n  \"timestamp\": {},\n  \"hash\": {},\n  \"proof_name\": \"{}\",\n  \"category\": \"{}\",\n  \"function\": \"{}\",\n  \"strength\": \"{}\",\n  \"verified\": {}\n}}\n",
        escape_json(result_tag),
        solver,
        timestamp,
        hash_str,
        escape_json(&cert.rule_name),
        escape_json(&cert.category),
        escape_json(&cert.function_name),
        escape_json(&cert.strength),
        cert.verified,
    );
    fs::write(path, body)
}

/// Emit `.smt2` and `.cert` files for every certificate in `certs` under
/// `<out_dir>/<Category>/<rule_name>.{smt2,cert}`.
///
/// Returns a summary count. Errors creating directories or writing files
/// are propagated to the caller.
pub fn emit_proof_files(
    out_dir: &Path,
    certs: &[CodegenCertificate],
) -> std::io::Result<EmitSummary> {
    if certs.is_empty() {
        return Ok(EmitSummary::default());
    }

    fs::create_dir_all(out_dir)?;

    // Build the obligation index once per invocation; ProofDatabase::new()
    // is pure construction and not cached, so we avoid rebuilding it per
    // certificate (several thousand entries).
    let db = ProofDatabase::new();
    let index = build_obligation_index(&db);
    let all = db.all();

    let mut summary = EmitSummary::default();
    // Dedup on (category_dir, rule_name) so duplicate certs (same rule
    // applied in multiple functions) do not thrash the filesystem.
    let mut seen: std::collections::HashSet<(String, String)> =
        std::collections::HashSet::new();

    for cert in certs {
        let (category_dir, obligation_idx) = match index.get(&cert.rule_name) {
            Some((idx, cat)) => (sanitize_path(cat), Some(*idx)),
            None => {
                // No obligation in the database — fall back to the string
                // category from the codegen certificate so we still
                // organise the file correctly.
                (sanitize_path(&cert.category), None)
            }
        };

        let file_stem = sanitize_path(&cert.rule_name);
        let key = (category_dir.clone(), file_stem.clone());
        if !seen.insert(key) {
            continue;
        }

        let dir = out_dir.join(&category_dir);
        fs::create_dir_all(&dir)?;

        let smt2_path: PathBuf = dir.join(format!("{}.smt2", file_stem));
        let cert_path: PathBuf = dir.join(format!("{}.cert", file_stem));

        let (smt2_hash, result_tag) = match obligation_idx {
            Some(idx) => {
                let smt2 = serialize_to_smt2(&all[idx].obligation);
                let hash = fnv1a_64(smt2.as_bytes());
                fs::write(&smt2_path, &smt2)?;
                summary.smt2_written += 1;
                let tag = if cert.verified { "verified" } else { "failed" };
                (Some(hash), tag)
            }
            None => {
                // No SMT available — record an eval-only certificate so
                // auditors can see the rule was attempted even without
                // a matching obligation in the database.
                (None, "eval-only")
            }
        };

        write_cert_file(&cert_path, cert, smt2_hash, result_tag)?;
        summary.cert_written += 1;

        if obligation_idx.is_none() {
            summary.skipped_no_obligation += 1;
        }
    }

    Ok(summary)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_path_handles_special_chars() {
        assert_eq!(sanitize_path("foo_bar"), "foo_bar");
        assert_eq!(sanitize_path("Floating-Point"), "Floating-Point");
        assert_eq!(sanitize_path("a/b"), "a_b");
        assert_eq!(sanitize_path(""), "unknown");
    }

    #[test]
    fn fnv1a_is_stable() {
        // Known FNV-1a-64 value for the empty string.
        assert_eq!(fnv1a_64(b""), 0xcbf2_9ce4_8422_2325);
        // Deterministic across calls.
        assert_eq!(fnv1a_64(b"hello"), fnv1a_64(b"hello"));
        assert_ne!(fnv1a_64(b"hello"), fnv1a_64(b"world"));
    }

    #[test]
    fn empty_certs_produces_no_files() {
        let tmp = std::env::temp_dir().join(format!(
            "llvm2_cli_emit_proofs_empty_{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&tmp);
        let s = emit_proof_files(&tmp, &[]).expect("empty ok");
        assert_eq!(s.smt2_written, 0);
        assert_eq!(s.cert_written, 0);
        let _ = fs::remove_dir_all(&tmp);
    }
}
