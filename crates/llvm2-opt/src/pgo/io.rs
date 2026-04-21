// llvm2-opt/pgo/io.rs - .profdata writer + reader
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// v0 serializes as JSON via serde_json. When we upgrade to a binary wire
// format, bump `PROFDATA_VERSION` in schema.rs; readers gated on the
// version will still accept the older JSON for backwards compat.

//! I/O helpers for reading and writing `.profdata` files.

use std::fs;
use std::io::{self, Read, Write};
use std::path::Path;

use crate::cache::CACHE_KEY_VERSION;

use super::schema::{PROFDATA_MAGIC, PROFDATA_VERSION, ProfData};

/// Errors that can be produced while reading or writing a `.profdata` file.
#[derive(Debug, thiserror::Error)]
pub enum ProfDataError {
    /// Filesystem or I/O failure.
    #[error("profdata I/O error: {0}")]
    Io(#[from] io::Error),
    /// JSON (de)serialization failure.
    #[error("profdata serialization error: {0}")]
    Serde(#[from] serde_json::Error),
    /// File header did not start with [`PROFDATA_MAGIC`].
    #[error("profdata magic mismatch: expected {expected:?}, found {found:?}")]
    BadMagic {
        /// Expected magic string.
        expected: &'static str,
        /// Actual magic value found in the file.
        found: String,
    },
    /// Schema version is newer than this reader understands.
    #[error(
        "profdata schema version too new: file={file}, reader={reader}"
    )]
    VersionTooNew {
        /// Version recorded in the file.
        file: u32,
        /// Version supported by this reader.
        reader: u32,
    },
    /// [`crate::cache::CACHE_KEY_VERSION`] mismatch — the recorded
    /// `module_hash` is not comparable to hashes produced by this build.
    #[error(
        "profdata cache_key_version mismatch: file={file}, reader={reader}"
    )]
    CacheKeyVersionMismatch {
        /// `CACHE_KEY_VERSION` stored in the file.
        file: u32,
        /// Current [`crate::cache::CACHE_KEY_VERSION`].
        reader: u32,
    },
    /// The stored `module_hash` did not match the hash of the module
    /// currently being compiled.
    #[error(
        "profdata module_hash stale: file={file_hash}, module={module_hash}"
    )]
    StaleHash {
        /// Hex string recorded in the file.
        file_hash: String,
        /// Hex string of the module currently being compiled.
        module_hash: String,
    },
}

/// Serialize a [`ProfData`] to a JSON byte vector.
///
/// The byte vector is suitable for direct filesystem writes or for
/// embedding in another container.
pub fn encode(profile: &ProfData) -> Result<Vec<u8>, ProfDataError> {
    // Pretty-print so `.profdata` is human-inspectable. JSON is the v0
    // format; a binary format is a follow-up.
    Ok(serde_json::to_vec_pretty(profile)?)
}

/// Deserialize a [`ProfData`] from raw bytes, validating the header.
///
/// Returns a [`ProfDataError`] if the magic, schema version, or
/// cache-key version are incompatible.
pub fn decode(bytes: &[u8]) -> Result<ProfData, ProfDataError> {
    let profile: ProfData = serde_json::from_slice(bytes)?;
    validate_header(&profile)?;
    Ok(profile)
}

/// Write a [`ProfData`] to `path`, overwriting any existing file.
pub fn write_to_path(profile: &ProfData, path: &Path) -> Result<(), ProfDataError> {
    let bytes = encode(profile)?;
    let mut f = fs::File::create(path)?;
    f.write_all(&bytes)?;
    f.sync_all()?;
    Ok(())
}

/// Read a [`ProfData`] from `path`.
pub fn read_from_path(path: &Path) -> Result<ProfData, ProfDataError> {
    let mut f = fs::File::open(path)?;
    let mut bytes = Vec::new();
    f.read_to_end(&mut bytes)?;
    decode(&bytes)
}

/// Verify a decoded [`ProfData`] against the hash of the module
/// currently being compiled.
///
/// Returns `Ok(())` if the hashes match (i.e., the profile is fresh).
/// Returns [`ProfDataError::StaleHash`] otherwise.
pub fn enforce_fresh(
    profile: &ProfData,
    current_module_hash: u128,
) -> Result<(), ProfDataError> {
    let expected = format!("{:032x}", current_module_hash);
    if profile.module_hash == expected {
        Ok(())
    } else {
        Err(ProfDataError::StaleHash {
            file_hash: profile.module_hash.clone(),
            module_hash: expected,
        })
    }
}

fn validate_header(profile: &ProfData) -> Result<(), ProfDataError> {
    if profile.magic != PROFDATA_MAGIC {
        return Err(ProfDataError::BadMagic {
            expected: PROFDATA_MAGIC,
            found: profile.magic.clone(),
        });
    }
    if profile.version > PROFDATA_VERSION {
        return Err(ProfDataError::VersionTooNew {
            file: profile.version,
            reader: PROFDATA_VERSION,
        });
    }
    if profile.cache_key_version != CACHE_KEY_VERSION {
        return Err(ProfDataError::CacheKeyVersionMismatch {
            file: profile.cache_key_version,
            reader: CACHE_KEY_VERSION,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pgo::schema::{BlockProfile, FunctionProfile};

    fn sample_profile() -> ProfData {
        let mut p = ProfData::new(0xdead_beef_cafe_babe_0123_4567_89ab_cdef);
        p.target_triple = "aarch64-apple-darwin".into();
        p.opt_level = "O2".into();

        let mut f = FunctionProfile::new("bfs_step");
        f.call_count = 10_000;
        f.blocks = vec![
            BlockProfile::new(0, 10_000),
            BlockProfile::new(1, 9_750),
            BlockProfile::new(2, 250),
            BlockProfile::new(3, 0),
        ];
        p.functions.push(f);

        let g = FunctionProfile::new("cold_helper");
        p.functions.push(g);
        p
    }

    #[test]
    fn round_trip_bytes_match_structurally() {
        let p = sample_profile();
        let bytes = encode(&p).unwrap();
        let q = decode(&bytes).unwrap();
        assert_eq!(p, q, "encode/decode round trip must be lossless");
        // Per-block sanity
        let f = q.function("bfs_step").unwrap();
        assert_eq!(f.call_count, 10_000);
        assert_eq!(f.block_hits(0), 10_000);
        assert_eq!(f.block_hits(1), 9_750);
        assert_eq!(f.block_hits(3), 0);
        assert_eq!(f.block_hits(42), 0, "missing blocks read back as 0");
    }

    #[test]
    fn round_trip_file_path() {
        let p = sample_profile();
        let tmp = std::env::temp_dir().join(format!(
            "llvm2_profdata_test_{}.profdata",
            std::process::id()
        ));
        write_to_path(&p, &tmp).unwrap();
        let q = read_from_path(&tmp).unwrap();
        assert_eq!(p, q);
        let _ = fs::remove_file(&tmp);
    }

    #[test]
    fn decode_rejects_bad_magic() {
        let mut p = sample_profile();
        p.magic = "WRONG".to_string();
        let bytes = encode(&p).unwrap();
        match decode(&bytes) {
            Err(ProfDataError::BadMagic { .. }) => {}
            other => panic!("expected BadMagic, got {:?}", other),
        }
    }

    #[test]
    fn decode_rejects_future_version() {
        let mut p = sample_profile();
        p.version = PROFDATA_VERSION + 1;
        let bytes = encode(&p).unwrap();
        match decode(&bytes) {
            Err(ProfDataError::VersionTooNew { file, reader }) => {
                assert_eq!(file, PROFDATA_VERSION + 1);
                assert_eq!(reader, PROFDATA_VERSION);
            }
            other => panic!("expected VersionTooNew, got {:?}", other),
        }
    }

    #[test]
    fn decode_rejects_cache_key_version_mismatch() {
        let mut p = sample_profile();
        p.cache_key_version = CACHE_KEY_VERSION.wrapping_add(7);
        let bytes = encode(&p).unwrap();
        match decode(&bytes) {
            Err(ProfDataError::CacheKeyVersionMismatch { .. }) => {}
            other => panic!("expected CacheKeyVersionMismatch, got {:?}", other),
        }
    }

    #[test]
    fn enforce_fresh_accepts_matching_hash() {
        let p = sample_profile();
        let h = p.module_hash_u128().unwrap();
        enforce_fresh(&p, h).unwrap();
    }

    #[test]
    fn enforce_fresh_rejects_mismatched_hash() {
        let p = sample_profile();
        let h = p.module_hash_u128().unwrap();
        match enforce_fresh(&p, h.wrapping_add(1)) {
            Err(ProfDataError::StaleHash { .. }) => {}
            other => panic!("expected StaleHash, got {:?}", other),
        }
    }
}
