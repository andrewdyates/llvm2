// llvm2-ir - Provenance tracking: tMIR-to-binary offset mapping
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Reference: designs/2026-04-13-debugging-transparency.md (Provenance Tracking)
//
// This module maintains source-to-binary mappings through every compilation
// stage: ISel creates tMIR->MachIR mappings, optimization passes update them,
// and encoding adds the final binary offsets. The transitive closure gives
// tMIR->binary offset for the full chain.

//! Provenance tracking infrastructure for LLVM2.
//!
//! Maintains a bidirectional mapping from tMIR source instructions to machine
//! instructions to final binary offsets. Every compilation stage participates:
//!
//! - **ISel (llvm2-lower):** Records initial tMIR -> MachIR mapping via
//!   [`ProvenanceMap::record_lowering`].
//! - **Optimization passes (llvm2-opt):** Update mappings when instructions are
//!   replaced, merged, deleted, or created. See [`ProvenanceMap::record_replacement`],
//!   [`ProvenanceMap::record_merge`], [`ProvenanceMap::record_deletion`],
//!   [`ProvenanceMap::record_creation`].
//! - **Encoding (llvm2-codegen):** Records MachIR -> binary offset via
//!   [`ProvenanceMap::record_encoding`].
//! - **Query:** After all stages, call [`ProvenanceMap::build_transitive`] then
//!   use [`ProvenanceMap::query_offset`] or [`ProvenanceMap::query_source`].
//!
//! # Pass Update Rules
//!
//! | Pass action | Method | Provenance effect |
//! |------------|--------|-------------------|
//! | Replace A with B | `record_replacement` | `provenance[B] = provenance[A]` |
//! | Merge A,B into C | `record_merge` | `provenance[C] = union(provenance[A], provenance[B])` |
//! | Delete A (DCE) | `record_deletion` | Mark as OptimizedAway with justification |
//! | Create new inst | `record_creation` | Mark as CompilerGenerated |

use std::collections::HashMap;

use serde::Serialize;

use crate::types::InstId;

// ---------------------------------------------------------------------------
// TmirInstId — tMIR instruction identifier
// ---------------------------------------------------------------------------

/// Identifier for a tMIR instruction (source-level).
///
/// This corresponds to an instruction index in the tMIR function being compiled.
/// It is deliberately a separate type from `tmir::ValueId` to keep the
/// provenance system self-contained within llvm2-ir (no dependency on tMIR
/// for the core tracking infrastructure).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
pub struct TmirInstId(pub u32);

impl core::fmt::Display for TmirInstId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "tmir{}", self.0)
    }
}

// ---------------------------------------------------------------------------
// PassId — identifies which compiler pass touched an instruction
// ---------------------------------------------------------------------------

/// Identifier for a compiler pass (used in transformation chains).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
pub struct PassId(pub String);

impl PassId {
    /// Create a new PassId.
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    /// Get the pass name.
    pub fn name(&self) -> &str {
        &self.0
    }
}

impl core::fmt::Display for PassId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.0)
    }
}

// ---------------------------------------------------------------------------
// ProvenanceStatus — lifecycle state of a machine instruction's provenance
// ---------------------------------------------------------------------------

/// Status of a provenance entry — tracks whether the instruction is still
/// live or has been removed/created by the compiler.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProvenanceStatus {
    /// Instruction is active and maps to source tMIR instruction(s).
    Active,
    /// Instruction was removed by an optimization pass.
    OptimizedAway {
        /// Which pass removed it.
        pass: PassId,
        /// Why it was safe to remove.
        justification: String,
    },
    /// Instruction was created by the compiler (no direct tMIR source).
    /// Examples: spill code, frame setup, materialized constants.
    CompilerGenerated {
        /// Which pass or phase created it.
        pass: PassId,
        /// Why it was created.
        reason: String,
    },
}

// ---------------------------------------------------------------------------
// TransformRecord — one entry in the transformation chain
// ---------------------------------------------------------------------------

/// A record of one transformation applied to an instruction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransformRecord {
    /// Which pass applied this transformation.
    pub pass: PassId,
    /// What kind of transformation.
    pub kind: TransformKind,
}

/// The kind of transformation applied to an instruction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransformKind {
    /// Initial lowering from tMIR to MachIR.
    Lowered,
    /// Replaced by a different instruction (1:1).
    Replaced {
        /// The old instruction that was replaced.
        old: InstId,
    },
    /// Merged from multiple source instructions.
    Merged {
        /// The source instructions that were merged.
        sources: Vec<InstId>,
    },
    /// Instruction survived a pass unchanged.
    Survived,
    /// Binary encoding assigned an offset.
    Encoded {
        /// The binary offset assigned.
        offset: u32,
    },
}

// ---------------------------------------------------------------------------
// ProvenanceEntry — per-instruction provenance metadata
// ---------------------------------------------------------------------------

/// Full provenance metadata for a single machine instruction.
///
/// Tracks the instruction's origin (which tMIR instruction(s) it came from),
/// its transformation chain (which passes touched it), and its current status.
#[derive(Debug, Clone)]
pub struct ProvenanceEntry {
    /// Source tMIR instruction(s) this machine instruction originated from.
    /// Usually one, but merges can produce entries with multiple sources.
    pub tmir_origins: Vec<TmirInstId>,
    /// Ordered record of transformations applied to this instruction.
    pub transforms: Vec<TransformRecord>,
    /// Current lifecycle status.
    pub status: ProvenanceStatus,
}

impl ProvenanceEntry {
    /// Create a new entry originating from a single tMIR instruction.
    pub fn from_lowering(tmir_id: TmirInstId, pass: PassId) -> Self {
        Self {
            tmir_origins: vec![tmir_id],
            transforms: vec![TransformRecord {
                pass,
                kind: TransformKind::Lowered,
            }],
            status: ProvenanceStatus::Active,
        }
    }

    /// Create a compiler-generated entry (no tMIR source).
    pub fn compiler_generated(pass: PassId, reason: String) -> Self {
        Self {
            tmir_origins: Vec::new(),
            transforms: Vec::new(),
            status: ProvenanceStatus::CompilerGenerated { pass, reason },
        }
    }

    /// Returns true if this instruction is still active.
    pub fn is_active(&self) -> bool {
        matches!(self.status, ProvenanceStatus::Active)
    }

    /// Returns true if this instruction was optimized away.
    pub fn is_optimized_away(&self) -> bool {
        matches!(self.status, ProvenanceStatus::OptimizedAway { .. })
    }

    /// Returns true if this instruction was compiler-generated.
    pub fn is_compiler_generated(&self) -> bool {
        matches!(self.status, ProvenanceStatus::CompilerGenerated { .. })
    }
}

// ---------------------------------------------------------------------------
// ProvenanceMap — the core tracking structure
// ---------------------------------------------------------------------------

/// Source-to-binary mapping maintained through every compilation stage.
///
/// # Usage
///
/// ```text
/// // 1. ISel creates initial mapping:
/// provenance.record_lowering(tmir_id, &[mach_id1, mach_id2], pass);
///
/// // 2. Optimization passes update:
/// provenance.record_replacement(old_id, new_id, pass);   // 1:1 replace
/// provenance.record_merge(&[a, b], merged, pass);        // N:1 merge
/// provenance.record_deletion(dead_id, pass, "unused");   // DCE
/// provenance.record_creation(new_id, pass, "spill");     // materialization
///
/// // 3. Encoding adds offsets:
/// provenance.record_encoding(mach_id, 0x48);
///
/// // 4. Build transitive closure and query:
/// provenance.build_transitive();
/// let offsets = provenance.query_offset(tmir_id);   // tMIR -> [binary offsets]
/// let sources = provenance.query_source(0x48);       // offset -> [tMIR ids]
/// ```
#[derive(Debug, Clone)]
pub struct ProvenanceMap {
    /// tMIR instruction -> MachIR instructions (1:N from ISel).
    tmir_to_mach: HashMap<TmirInstId, Vec<InstId>>,
    /// MachIR instruction -> binary offset (1:1 from encoding).
    mach_to_offset: HashMap<InstId, u32>,
    /// Transitive: tMIR -> binary offsets (built by `build_transitive`).
    tmir_to_offset: HashMap<TmirInstId, Vec<u32>>,
    /// Reverse: binary offset -> tMIR instructions (built by `build_transitive`).
    offset_to_tmir: HashMap<u32, Vec<TmirInstId>>,
    /// Per-instruction provenance metadata.
    entries: HashMap<InstId, ProvenanceEntry>,
}

impl ProvenanceMap {
    /// Create an empty provenance map.
    pub fn new() -> Self {
        Self {
            tmir_to_mach: HashMap::new(),
            mach_to_offset: HashMap::new(),
            tmir_to_offset: HashMap::new(),
            offset_to_tmir: HashMap::new(),
            entries: HashMap::new(),
        }
    }

    // -- Stage 1: ISel (tMIR -> MachIR) --

    /// Record that a tMIR instruction was lowered to one or more machine
    /// instructions during instruction selection.
    ///
    /// This is the initial mapping created by ISel. Each tMIR instruction
    /// may produce multiple machine instructions (e.g., a 64-bit multiply
    /// that expands to multiple AArch64 instructions).
    pub fn record_lowering(
        &mut self,
        tmir_id: TmirInstId,
        mach_ids: &[InstId],
        pass: PassId,
    ) {
        self.tmir_to_mach
            .entry(tmir_id)
            .or_default()
            .extend_from_slice(mach_ids);

        for &mach_id in mach_ids {
            self.entries
                .insert(mach_id, ProvenanceEntry::from_lowering(tmir_id, pass.clone()));
        }
    }

    // -- Stage 2: Optimization pass updates --

    /// Record that an optimization pass replaced instruction `old` with `new`.
    ///
    /// The new instruction inherits all provenance from the old one.
    /// Rule: `provenance[new] = provenance[old]`
    pub fn record_replacement(&mut self, old: InstId, new: InstId, pass: PassId) {
        // Transfer provenance entry from old to new.
        if let Some(mut entry) = self.entries.remove(&old) {
            entry.transforms.push(TransformRecord {
                pass: pass.clone(),
                kind: TransformKind::Replaced { old },
            });
            self.entries.insert(new, entry);
        }

        // Update tmir_to_mach: replace old with new in all mappings.
        for mach_ids in self.tmir_to_mach.values_mut() {
            for id in mach_ids.iter_mut() {
                if *id == old {
                    *id = new;
                }
            }
        }
    }

    /// Record that multiple instructions were merged into one.
    ///
    /// The merged instruction inherits provenance from all sources.
    /// Rule: `provenance[merged] = union(provenance[sources])`
    pub fn record_merge(&mut self, sources: &[InstId], merged: InstId, pass: PassId) {
        // Collect all tMIR origins from all source entries.
        let mut all_origins: Vec<TmirInstId> = Vec::new();
        let mut all_transforms: Vec<TransformRecord> = Vec::new();

        for &src in sources {
            if let Some(entry) = self.entries.remove(&src) {
                for origin in &entry.tmir_origins {
                    if !all_origins.contains(origin) {
                        all_origins.push(*origin);
                    }
                }
                all_transforms.extend(entry.transforms);
            }
        }

        // Create merged entry.
        let mut merged_entry = ProvenanceEntry {
            tmir_origins: all_origins,
            transforms: all_transforms,
            status: ProvenanceStatus::Active,
        };
        merged_entry.transforms.push(TransformRecord {
            pass: pass.clone(),
            kind: TransformKind::Merged {
                sources: sources.to_vec(),
            },
        });
        self.entries.insert(merged, merged_entry);

        // Update tmir_to_mach: replace all sources with merged.
        for mach_ids in self.tmir_to_mach.values_mut() {
            let mut found = false;
            mach_ids.retain(|id| {
                if sources.contains(id) {
                    if !found {
                        found = true;
                        // Keep this slot, will be overwritten below.
                        true
                    } else {
                        false
                    }
                } else {
                    true
                }
            });
            // Replace the first occurrence of a source with merged.
            for id in mach_ids.iter_mut() {
                if sources.contains(id) {
                    *id = merged;
                    break;
                }
            }
        }
    }

    /// Record that an instruction was deleted by an optimization pass (DCE).
    ///
    /// The instruction is marked as OptimizedAway with a justification.
    pub fn record_deletion(
        &mut self,
        inst: InstId,
        pass: PassId,
        justification: impl Into<String>,
    ) {
        if let Some(entry) = self.entries.get_mut(&inst) {
            entry.status = ProvenanceStatus::OptimizedAway {
                pass,
                justification: justification.into(),
            };
        }
        // Note: we keep the entry in the map so queries can report
        // "this tMIR instruction was optimized away" rather than silence.
    }

    /// Record that a new instruction was created by the compiler (materialization,
    /// spill code, frame setup, etc.) with no direct tMIR source.
    pub fn record_creation(
        &mut self,
        inst: InstId,
        pass: PassId,
        reason: impl Into<String>,
    ) {
        self.entries.insert(
            inst,
            ProvenanceEntry::compiler_generated(pass, reason.into()),
        );
    }

    // -- Stage 3: Encoding (MachIR -> binary offset) --

    /// Record that a machine instruction was encoded at a binary offset.
    pub fn record_encoding(&mut self, inst: InstId, offset: u32) {
        self.mach_to_offset.insert(inst, offset);

        if let Some(entry) = self.entries.get_mut(&inst) {
            entry.transforms.push(TransformRecord {
                pass: PassId::new("encoding"),
                kind: TransformKind::Encoded { offset },
            });
        }
    }

    // -- Stage 4: Build transitive closure --

    /// Build the transitive tMIR -> binary offset mapping.
    ///
    /// Must be called after all encoding is complete. Populates
    /// `tmir_to_offset` and `offset_to_tmir` for query use.
    pub fn build_transitive(&mut self) {
        self.tmir_to_offset.clear();
        self.offset_to_tmir.clear();

        for (&tmir_id, mach_ids) in &self.tmir_to_mach {
            let mut offsets = Vec::new();
            for &mach_id in mach_ids {
                // Only include active instructions with binary offsets.
                if let Some(&offset) = self.mach_to_offset.get(&mach_id)
                    && self
                        .entries
                        .get(&mach_id)
                        .is_none_or(|e| e.is_active())
                    {
                        offsets.push(offset);
                        self.offset_to_tmir
                            .entry(offset)
                            .or_default()
                            .push(tmir_id);
                    }
            }
            offsets.sort_unstable();
            offsets.dedup();
            if !offsets.is_empty() {
                self.tmir_to_offset.insert(tmir_id, offsets);
            }
        }

        // Deduplicate reverse mapping entries.
        for tmir_ids in self.offset_to_tmir.values_mut() {
            tmir_ids.sort_unstable();
            tmir_ids.dedup();
        }
    }

    // -- Query methods --

    /// Query the binary offsets for a tMIR instruction.
    ///
    /// Returns `None` if the tMIR instruction has no binary representation
    /// (e.g., it was optimized away entirely).
    pub fn query_offset(&self, tmir_id: TmirInstId) -> Option<&[u32]> {
        self.tmir_to_offset.get(&tmir_id).map(|v| v.as_slice())
    }

    /// Query which tMIR instructions contributed to a binary offset.
    ///
    /// Returns `None` if the offset is compiler-generated with no tMIR source.
    pub fn query_source(&self, offset: u32) -> Option<&[TmirInstId]> {
        self.offset_to_tmir.get(&offset).map(|v| v.as_slice())
    }

    /// Get the provenance entry for a machine instruction.
    pub fn get_entry(&self, inst: InstId) -> Option<&ProvenanceEntry> {
        self.entries.get(&inst)
    }

    /// Get the binary offset for a machine instruction.
    pub fn get_offset(&self, inst: InstId) -> Option<u32> {
        self.mach_to_offset.get(&inst).copied()
    }

    /// Get all machine instructions that a tMIR instruction was lowered to.
    pub fn get_mach_insts(&self, tmir_id: TmirInstId) -> Option<&[InstId]> {
        self.tmir_to_mach.get(&tmir_id).map(|v| v.as_slice())
    }

    /// Returns the number of tMIR instructions tracked.
    pub fn num_tmir_entries(&self) -> usize {
        self.tmir_to_mach.len()
    }

    /// Returns the number of machine instructions with provenance entries.
    pub fn num_mach_entries(&self) -> usize {
        self.entries.len()
    }

    /// Returns the number of encoded machine instructions.
    pub fn num_encoded(&self) -> usize {
        self.mach_to_offset.len()
    }

    /// Returns an iterator over all tMIR instructions that were optimized away.
    pub fn optimized_away(&self) -> Vec<(InstId, &ProvenanceEntry)> {
        self.entries
            .iter()
            .filter(|(_, e)| e.is_optimized_away())
            .map(|(&id, e)| (id, e))
            .collect()
    }

    /// Returns an iterator over all compiler-generated instructions.
    pub fn compiler_generated(&self) -> Vec<(InstId, &ProvenanceEntry)> {
        self.entries
            .iter()
            .filter(|(_, e)| e.is_compiler_generated())
            .map(|(&id, e)| (id, e))
            .collect()
    }

    /// Returns a summary of provenance statistics.
    pub fn stats(&self) -> ProvenanceStats {
        let mut active = 0u32;
        let mut optimized_away = 0u32;
        let mut compiler_generated = 0u32;

        for entry in self.entries.values() {
            match entry.status {
                ProvenanceStatus::Active => active += 1,
                ProvenanceStatus::OptimizedAway { .. } => optimized_away += 1,
                ProvenanceStatus::CompilerGenerated { .. } => compiler_generated += 1,
            }
        }

        ProvenanceStats {
            tmir_instructions: self.tmir_to_mach.len() as u32,
            mach_instructions: self.entries.len() as u32,
            active,
            optimized_away,
            compiler_generated,
            encoded: self.mach_to_offset.len() as u32,
            transitive_mappings: self.tmir_to_offset.len() as u32,
        }
    }
}

impl Default for ProvenanceMap {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ProvenanceStats — summary statistics
// ---------------------------------------------------------------------------

/// Summary statistics for a provenance map.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProvenanceStats {
    /// Number of tMIR instructions tracked.
    pub tmir_instructions: u32,
    /// Total number of machine instructions with provenance entries.
    pub mach_instructions: u32,
    /// Number of active (live) machine instructions.
    pub active: u32,
    /// Number of machine instructions optimized away.
    pub optimized_away: u32,
    /// Number of compiler-generated instructions.
    pub compiler_generated: u32,
    /// Number of machine instructions with binary offsets.
    pub encoded: u32,
    /// Number of tMIR instructions with transitive binary mappings.
    pub transitive_mappings: u32,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn isel_pass() -> PassId {
        PassId::new("isel")
    }

    fn dce_pass() -> PassId {
        PassId::new("dce")
    }

    fn peephole_pass() -> PassId {
        PassId::new("peephole")
    }

    fn regalloc_pass() -> PassId {
        PassId::new("regalloc")
    }

    // -- TmirInstId tests --

    #[test]
    fn tmir_inst_id_display() {
        assert_eq!(format!("{}", TmirInstId(0)), "tmir0");
        assert_eq!(format!("{}", TmirInstId(42)), "tmir42");
    }

    #[test]
    fn tmir_inst_id_equality_and_ordering() {
        assert_eq!(TmirInstId(5), TmirInstId(5));
        assert_ne!(TmirInstId(5), TmirInstId(6));
        assert!(TmirInstId(0) < TmirInstId(1));
    }

    #[test]
    fn tmir_inst_id_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(TmirInstId(0));
        set.insert(TmirInstId(0)); // duplicate
        set.insert(TmirInstId(1));
        assert_eq!(set.len(), 2);
    }

    // -- PassId tests --

    #[test]
    fn pass_id_creation_and_display() {
        let p = PassId::new("dce");
        assert_eq!(p.name(), "dce");
        assert_eq!(format!("{}", p), "dce");
    }

    // -- ProvenanceEntry tests --

    #[test]
    fn provenance_entry_from_lowering() {
        let entry = ProvenanceEntry::from_lowering(TmirInstId(10), isel_pass());
        assert!(entry.is_active());
        assert!(!entry.is_optimized_away());
        assert!(!entry.is_compiler_generated());
        assert_eq!(entry.tmir_origins.len(), 1);
        assert_eq!(entry.tmir_origins[0], TmirInstId(10));
        assert_eq!(entry.transforms.len(), 1);
        assert_eq!(entry.transforms[0].kind, TransformKind::Lowered);
    }

    #[test]
    fn provenance_entry_compiler_generated() {
        let entry = ProvenanceEntry::compiler_generated(
            regalloc_pass(),
            "spill reload".to_string(),
        );
        assert!(!entry.is_active());
        assert!(!entry.is_optimized_away());
        assert!(entry.is_compiler_generated());
        assert!(entry.tmir_origins.is_empty());
    }

    // -- ProvenanceMap: basic lowering --

    #[test]
    fn record_lowering_single_inst() {
        let mut map = ProvenanceMap::new();
        let tmir = TmirInstId(0);
        let mach = InstId(0);

        map.record_lowering(tmir, &[mach], isel_pass());

        assert_eq!(map.num_tmir_entries(), 1);
        assert_eq!(map.num_mach_entries(), 1);
        assert_eq!(map.get_mach_insts(tmir), Some(&[mach][..]));

        let entry = map.get_entry(mach).unwrap();
        assert!(entry.is_active());
        assert_eq!(entry.tmir_origins, vec![tmir]);
    }

    #[test]
    fn record_lowering_one_to_many() {
        let mut map = ProvenanceMap::new();
        let tmir = TmirInstId(5);
        let mach_ids = [InstId(10), InstId(11), InstId(12)];

        map.record_lowering(tmir, &mach_ids, isel_pass());

        assert_eq!(map.num_tmir_entries(), 1);
        assert_eq!(map.num_mach_entries(), 3);
        assert_eq!(map.get_mach_insts(tmir).unwrap().len(), 3);
    }

    #[test]
    fn record_lowering_multiple_tmir() {
        let mut map = ProvenanceMap::new();
        map.record_lowering(TmirInstId(0), &[InstId(0)], isel_pass());
        map.record_lowering(TmirInstId(1), &[InstId(1), InstId(2)], isel_pass());
        map.record_lowering(TmirInstId(2), &[InstId(3)], isel_pass());

        assert_eq!(map.num_tmir_entries(), 3);
        assert_eq!(map.num_mach_entries(), 4);
    }

    // -- ProvenanceMap: replacement --

    #[test]
    fn record_replacement_transfers_provenance() {
        let mut map = ProvenanceMap::new();
        let tmir = TmirInstId(0);
        let old = InstId(0);
        let new = InstId(1);

        map.record_lowering(tmir, &[old], isel_pass());
        map.record_replacement(old, new, peephole_pass());

        // Old entry should be gone.
        assert!(map.get_entry(old).is_none());

        // New entry should have the old's provenance.
        let entry = map.get_entry(new).unwrap();
        assert!(entry.is_active());
        assert_eq!(entry.tmir_origins, vec![tmir]);
        assert_eq!(entry.transforms.len(), 2); // lowered + replaced

        // tmir_to_mach should point to new.
        let mach_ids = map.get_mach_insts(tmir).unwrap();
        assert_eq!(mach_ids, &[new]);
    }

    // -- ProvenanceMap: merge --

    #[test]
    fn record_merge_combines_provenance() {
        let mut map = ProvenanceMap::new();
        let tmir_a = TmirInstId(0);
        let tmir_b = TmirInstId(1);
        let inst_a = InstId(0);
        let inst_b = InstId(1);
        let merged = InstId(2);

        map.record_lowering(tmir_a, &[inst_a], isel_pass());
        map.record_lowering(tmir_b, &[inst_b], isel_pass());
        map.record_merge(&[inst_a, inst_b], merged, peephole_pass());

        // Source entries should be removed.
        assert!(map.get_entry(inst_a).is_none());
        assert!(map.get_entry(inst_b).is_none());

        // Merged entry has both origins.
        let entry = map.get_entry(merged).unwrap();
        assert!(entry.is_active());
        assert_eq!(entry.tmir_origins.len(), 2);
        assert!(entry.tmir_origins.contains(&tmir_a));
        assert!(entry.tmir_origins.contains(&tmir_b));
    }

    // -- ProvenanceMap: deletion --

    #[test]
    fn record_deletion_marks_optimized_away() {
        let mut map = ProvenanceMap::new();
        let tmir = TmirInstId(0);
        let inst = InstId(0);

        map.record_lowering(tmir, &[inst], isel_pass());
        map.record_deletion(inst, dce_pass(), "result unused");

        let entry = map.get_entry(inst).unwrap();
        assert!(entry.is_optimized_away());
        assert!(!entry.is_active());
        assert_eq!(entry.tmir_origins, vec![tmir]);
    }

    // -- ProvenanceMap: creation --

    #[test]
    fn record_creation_marks_compiler_generated() {
        let mut map = ProvenanceMap::new();
        let inst = InstId(100);

        map.record_creation(inst, regalloc_pass(), "spill reload for vreg %5");

        let entry = map.get_entry(inst).unwrap();
        assert!(entry.is_compiler_generated());
        assert!(entry.tmir_origins.is_empty());
    }

    // -- ProvenanceMap: encoding --

    #[test]
    fn record_encoding_stores_offset() {
        let mut map = ProvenanceMap::new();
        let tmir = TmirInstId(0);
        let inst = InstId(0);

        map.record_lowering(tmir, &[inst], isel_pass());
        map.record_encoding(inst, 0x48);

        assert_eq!(map.get_offset(inst), Some(0x48));
        assert_eq!(map.num_encoded(), 1);

        // Check that encoding is in the transform chain.
        let entry = map.get_entry(inst).unwrap();
        let last = entry.transforms.last().unwrap();
        assert_eq!(last.kind, TransformKind::Encoded { offset: 0x48 });
    }

    // -- ProvenanceMap: transitive closure --

    #[test]
    fn build_transitive_basic() {
        let mut map = ProvenanceMap::new();

        // tmir0 -> [inst0, inst1], tmir1 -> [inst2]
        map.record_lowering(TmirInstId(0), &[InstId(0), InstId(1)], isel_pass());
        map.record_lowering(TmirInstId(1), &[InstId(2)], isel_pass());

        // Encode all.
        map.record_encoding(InstId(0), 0x00);
        map.record_encoding(InstId(1), 0x04);
        map.record_encoding(InstId(2), 0x08);

        map.build_transitive();

        // tmir0 -> [0x00, 0x04]
        let offsets = map.query_offset(TmirInstId(0)).unwrap();
        assert_eq!(offsets, &[0x00, 0x04]);

        // tmir1 -> [0x08]
        let offsets = map.query_offset(TmirInstId(1)).unwrap();
        assert_eq!(offsets, &[0x08]);

        // Reverse: 0x00 -> [tmir0]
        let sources = map.query_source(0x00).unwrap();
        assert_eq!(sources, &[TmirInstId(0)]);

        // Reverse: 0x08 -> [tmir1]
        let sources = map.query_source(0x08).unwrap();
        assert_eq!(sources, &[TmirInstId(1)]);
    }

    #[test]
    fn build_transitive_excludes_optimized_away() {
        let mut map = ProvenanceMap::new();

        map.record_lowering(TmirInstId(0), &[InstId(0), InstId(1)], isel_pass());

        // inst0 is encoded, inst1 is deleted.
        map.record_encoding(InstId(0), 0x00);
        map.record_deletion(InstId(1), dce_pass(), "dead code");

        map.build_transitive();

        // tmir0 -> [0x00] (inst1 was optimized away, no offset)
        let offsets = map.query_offset(TmirInstId(0)).unwrap();
        assert_eq!(offsets, &[0x00]);
    }

    #[test]
    fn build_transitive_after_replacement() {
        let mut map = ProvenanceMap::new();

        map.record_lowering(TmirInstId(0), &[InstId(0)], isel_pass());
        map.record_replacement(InstId(0), InstId(1), peephole_pass());
        map.record_encoding(InstId(1), 0x10);

        map.build_transitive();

        let offsets = map.query_offset(TmirInstId(0)).unwrap();
        assert_eq!(offsets, &[0x10]);
    }

    #[test]
    fn build_transitive_no_encoded_returns_none() {
        let mut map = ProvenanceMap::new();
        map.record_lowering(TmirInstId(0), &[InstId(0)], isel_pass());
        // No encoding step.
        map.build_transitive();

        assert!(map.query_offset(TmirInstId(0)).is_none());
    }

    // -- ProvenanceMap: full pipeline simulation --

    #[test]
    fn full_pipeline_simulation() {
        let mut map = ProvenanceMap::new();

        // ISel: 3 tMIR instructions -> 5 machine instructions.
        map.record_lowering(TmirInstId(0), &[InstId(0), InstId(1)], isel_pass());
        map.record_lowering(TmirInstId(1), &[InstId(2)], isel_pass());
        map.record_lowering(TmirInstId(2), &[InstId(3), InstId(4)], isel_pass());

        // Optimization: peephole replaces inst1 with inst5.
        map.record_replacement(InstId(1), InstId(5), peephole_pass());

        // Optimization: DCE removes inst3 (dead).
        map.record_deletion(InstId(3), dce_pass(), "result unused");

        // Regalloc creates spill code (compiler-generated).
        map.record_creation(InstId(6), regalloc_pass(), "spill vreg %7");

        // Encoding.
        map.record_encoding(InstId(0), 0x00);
        map.record_encoding(InstId(5), 0x04); // replaced inst1
        map.record_encoding(InstId(2), 0x08);
        map.record_encoding(InstId(4), 0x0C);
        map.record_encoding(InstId(6), 0x10); // spill code

        map.build_transitive();

        // tmir0 -> [0x00, 0x04] (inst0 + replaced inst1->inst5)
        let offsets = map.query_offset(TmirInstId(0)).unwrap();
        assert_eq!(offsets, &[0x00, 0x04]);

        // tmir1 -> [0x08]
        let offsets = map.query_offset(TmirInstId(1)).unwrap();
        assert_eq!(offsets, &[0x08]);

        // tmir2 -> [0x0C] (inst3 was DCE'd, only inst4 remains)
        let offsets = map.query_offset(TmirInstId(2)).unwrap();
        assert_eq!(offsets, &[0x0C]);

        // Spill code at 0x10 has no tMIR source.
        assert!(map.query_source(0x10).is_none());

        // Stats.
        let stats = map.stats();
        assert_eq!(stats.tmir_instructions, 3);
        assert_eq!(stats.active, 4); // inst0, inst5, inst2, inst4
        assert_eq!(stats.optimized_away, 1); // inst3
        assert_eq!(stats.compiler_generated, 1); // inst6
        assert_eq!(stats.encoded, 5);
        assert_eq!(stats.transitive_mappings, 3); // all 3 tMIR have offsets
    }

    // -- ProvenanceMap: optimized_away and compiler_generated queries --

    #[test]
    fn optimized_away_query() {
        let mut map = ProvenanceMap::new();
        map.record_lowering(TmirInstId(0), &[InstId(0)], isel_pass());
        map.record_lowering(TmirInstId(1), &[InstId(1)], isel_pass());
        map.record_deletion(InstId(0), dce_pass(), "dead");

        let dead = map.optimized_away();
        assert_eq!(dead.len(), 1);
        assert_eq!(dead[0].0, InstId(0));
    }

    #[test]
    fn compiler_generated_query() {
        let mut map = ProvenanceMap::new();
        map.record_creation(InstId(0), regalloc_pass(), "spill");
        map.record_creation(InstId(1), regalloc_pass(), "reload");

        let generated = map.compiler_generated();
        assert_eq!(generated.len(), 2);
    }

    // -- ProvenanceMap: stats --

    #[test]
    fn empty_map_stats() {
        let map = ProvenanceMap::new();
        let stats = map.stats();
        assert_eq!(stats.tmir_instructions, 0);
        assert_eq!(stats.mach_instructions, 0);
        assert_eq!(stats.active, 0);
        assert_eq!(stats.optimized_away, 0);
        assert_eq!(stats.compiler_generated, 0);
        assert_eq!(stats.encoded, 0);
        assert_eq!(stats.transitive_mappings, 0);
    }

    // -- ProvenanceMap: Default trait --

    #[test]
    fn default_is_empty() {
        let map = ProvenanceMap::default();
        assert_eq!(map.num_tmir_entries(), 0);
        assert_eq!(map.num_mach_entries(), 0);
        assert_eq!(map.num_encoded(), 0);
    }

    // -- Edge cases --

    #[test]
    fn replacement_of_nonexistent_is_silent() {
        let mut map = ProvenanceMap::new();
        // Replacing a non-existent instruction should not panic.
        map.record_replacement(InstId(99), InstId(100), peephole_pass());
        assert!(map.get_entry(InstId(99)).is_none());
        assert!(map.get_entry(InstId(100)).is_none());
    }

    #[test]
    fn deletion_of_nonexistent_is_silent() {
        let mut map = ProvenanceMap::new();
        // Deleting a non-existent instruction should not panic.
        map.record_deletion(InstId(99), dce_pass(), "not real");
        assert!(map.get_entry(InstId(99)).is_none());
    }

    #[test]
    fn encoding_without_lowering_still_stores_offset() {
        let mut map = ProvenanceMap::new();
        // Encoding an instruction without prior lowering (e.g., compiler-generated).
        map.record_creation(InstId(0), regalloc_pass(), "prologue");
        map.record_encoding(InstId(0), 0x00);

        assert_eq!(map.get_offset(InstId(0)), Some(0x00));
    }

    #[test]
    fn query_nonexistent_returns_none() {
        let map = ProvenanceMap::new();
        assert!(map.query_offset(TmirInstId(99)).is_none());
        assert!(map.query_source(0xDEAD).is_none());
        assert!(map.get_entry(InstId(99)).is_none());
        assert!(map.get_offset(InstId(99)).is_none());
        assert!(map.get_mach_insts(TmirInstId(99)).is_none());
    }

    #[test]
    fn duplicate_lowering_appends() {
        let mut map = ProvenanceMap::new();
        // Same tMIR instruction lowered in two calls (unusual but possible).
        map.record_lowering(TmirInstId(0), &[InstId(0)], isel_pass());
        map.record_lowering(TmirInstId(0), &[InstId(1)], isel_pass());

        let mach_ids = map.get_mach_insts(TmirInstId(0)).unwrap();
        assert_eq!(mach_ids.len(), 2);
    }

    #[test]
    fn merge_with_single_source_works() {
        let mut map = ProvenanceMap::new();
        map.record_lowering(TmirInstId(0), &[InstId(0)], isel_pass());
        map.record_merge(&[InstId(0)], InstId(1), peephole_pass());

        let entry = map.get_entry(InstId(1)).unwrap();
        assert!(entry.is_active());
        assert_eq!(entry.tmir_origins, vec![TmirInstId(0)]);
    }

    #[test]
    fn merge_deduplicates_origins() {
        let mut map = ProvenanceMap::new();
        // Two machine instructions from the same tMIR source.
        map.record_lowering(TmirInstId(0), &[InstId(0), InstId(1)], isel_pass());
        map.record_merge(&[InstId(0), InstId(1)], InstId(2), peephole_pass());

        let entry = map.get_entry(InstId(2)).unwrap();
        // Should have only one origin (deduplicated).
        assert_eq!(entry.tmir_origins.len(), 1);
        assert_eq!(entry.tmir_origins[0], TmirInstId(0));
    }
}
