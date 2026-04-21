// llvm2-codegen/constant_pool.rs - Float constant pool for x86-64
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Reference: Intel 64 and IA-32 Architectures SDM, Volume 1, Section 4.9
// Reference: System V AMD64 ABI, Section 3.5.1 (Data alignment)

//! Constant pool for x86-64 float constant materialization.
//!
//! On x86-64, there is no instruction to load a floating-point immediate
//! directly into an XMM register (unlike AArch64's FMOV). Instead, float
//! and double constants must be placed in a read-only data section (.rodata
//! for ELF, __DATA/__const or __TEXT/__literal for Mach-O) and loaded via
//! RIP-relative MOVSS/MOVSD instructions:
//!
//! ```text
//! movss xmm0, [rip + offset_to_float_constant]
//! movsd xmm1, [rip + offset_to_double_constant]
//! ```
//!
//! This module provides the [`ConstantPool`] data structure that:
//! 1. Collects f32/f64 constants during instruction selection
//! 2. Deduplicates identical constants (by bit pattern)
//! 3. Lays out entries with proper alignment (4 for f32, 8 for f64)
//! 4. Emits the raw bytes for the constant pool section

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Constant pool entry
// ---------------------------------------------------------------------------

/// A single entry in the constant pool.
#[derive(Debug, Clone)]
pub struct ConstantPoolEntry {
    /// Raw bytes of the constant (4 bytes for f32, 8 bytes for f64).
    pub data: Vec<u8>,
    /// Required alignment in bytes (4 for f32, 8 for f64).
    pub align: u32,
    /// Byte offset within the constant pool section (set by [`ConstantPool::layout`]).
    pub offset: u32,
}

// ---------------------------------------------------------------------------
// Constant pool
// ---------------------------------------------------------------------------

/// Dedup key marker: XOR'd with f64 bit patterns to separate the f32 and
/// f64 key spaces. This prevents collisions between, e.g., an f32 whose
/// bit pattern happens to match an f64's bits.
const F64_DEDUP_MARKER: u64 = 0x8000_0000_0000_0000;

/// A constant pool that collects float/double constants for emission in a
/// read-only data section.
///
/// Constants are deduplicated by bit pattern so that identical values share
/// the same pool entry.
#[derive(Debug, Clone)]
pub struct ConstantPool {
    /// Ordered list of constant pool entries.
    entries: Vec<ConstantPoolEntry>,
    /// Maps bit-pattern key -> entry index for deduplication.
    ///
    /// For f32: key = `val.to_bits() as u64`
    /// For f64: key = `val.to_bits() ^ F64_DEDUP_MARKER`
    dedup: HashMap<u64, usize>,
}

impl Default for ConstantPool {
    fn default() -> Self {
        Self::new()
    }
}

impl ConstantPool {
    /// Create a new empty constant pool.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            dedup: HashMap::new(),
        }
    }

    /// Add an f32 constant to the pool. Returns the entry index.
    ///
    /// If an f32 with the same bit pattern already exists, returns the
    /// existing entry index (deduplication).
    pub fn add_f32(&mut self, val: f32) -> usize {
        let key = val.to_bits() as u64;
        if let Some(&idx) = self.dedup.get(&key) {
            return idx;
        }
        let idx = self.entries.len();
        self.entries.push(ConstantPoolEntry {
            data: val.to_le_bytes().to_vec(),
            align: 4,
            offset: 0,
        });
        self.dedup.insert(key, idx);
        idx
    }

    /// Add an f64 constant to the pool. Returns the entry index.
    ///
    /// If an f64 with the same bit pattern already exists, returns the
    /// existing entry index (deduplication).
    pub fn add_f64(&mut self, val: f64) -> usize {
        let key = val.to_bits() ^ F64_DEDUP_MARKER;
        if let Some(&idx) = self.dedup.get(&key) {
            return idx;
        }
        let idx = self.entries.len();
        self.entries.push(ConstantPoolEntry {
            data: val.to_le_bytes().to_vec(),
            align: 8,
            offset: 0,
        });
        self.dedup.insert(key, idx);
        idx
    }

    /// Compute the layout of all entries, assigning byte offsets.
    ///
    /// Returns the total size of the constant pool in bytes.
    /// Must be called before [`emit`] or [`entry_offset`].
    pub fn layout(&mut self) -> u32 {
        let mut offset: u32 = 0;
        for entry in &mut self.entries {
            let align = entry.align;
            // Align up: offset = (offset + align - 1) & !(align - 1)
            offset = (offset + align - 1) & !(align - 1);
            entry.offset = offset;
            offset += entry.data.len() as u32;
        }
        offset
    }

    /// Emit the constant pool as a byte vector.
    ///
    /// The returned bytes include alignment padding (zero-filled) between
    /// entries. Call [`layout`] first to compute offsets.
    pub fn emit(&self) -> Vec<u8> {
        if self.entries.is_empty() {
            return Vec::new();
        }
        let total = self
            .entries
            .last()
            .map(|e| e.offset + e.data.len() as u32)
            .unwrap_or(0);
        let mut bytes = vec![0u8; total as usize];
        for entry in &self.entries {
            let start = entry.offset as usize;
            bytes[start..start + entry.data.len()].copy_from_slice(&entry.data);
        }
        bytes
    }

    /// Get the byte offset of entry `idx` within the constant pool.
    ///
    /// Panics if `idx` is out of range.
    pub fn entry_offset(&self, idx: usize) -> u32 {
        self.entries[idx].offset
    }

    /// Returns `true` if the constant pool has no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns the number of entries in the constant pool.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns a reference to the entries slice.
    pub fn entries(&self) -> &[ConstantPoolEntry] {
        &self.entries
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_pool() {
        let pool = ConstantPool::new();
        assert!(pool.is_empty());
        assert_eq!(pool.len(), 0);
        assert_eq!(pool.emit(), Vec::<u8>::new());
    }

    #[test]
    fn add_single_f32() {
        let mut pool = ConstantPool::new();
        let idx = pool.add_f32(3.14_f32);
        assert_eq!(idx, 0);
        assert_eq!(pool.len(), 1);
        assert!(!pool.is_empty());

        let total = pool.layout();
        assert_eq!(total, 4);
        assert_eq!(pool.entry_offset(0), 0);

        let bytes = pool.emit();
        assert_eq!(bytes, 3.14_f32.to_le_bytes().to_vec());
    }

    #[test]
    fn add_single_f64() {
        let mut pool = ConstantPool::new();
        let idx = pool.add_f64(2.718281828);
        assert_eq!(idx, 0);
        assert_eq!(pool.len(), 1);

        let total = pool.layout();
        assert_eq!(total, 8);
        assert_eq!(pool.entry_offset(0), 0);

        let bytes = pool.emit();
        assert_eq!(bytes, 2.718281828_f64.to_le_bytes().to_vec());
    }

    #[test]
    fn dedup_f32() {
        let mut pool = ConstantPool::new();
        let idx1 = pool.add_f32(1.0);
        let idx2 = pool.add_f32(1.0);
        assert_eq!(idx1, idx2);
        assert_eq!(pool.len(), 1);
    }

    #[test]
    fn dedup_f64() {
        let mut pool = ConstantPool::new();
        let idx1 = pool.add_f64(1.0);
        let idx2 = pool.add_f64(1.0);
        assert_eq!(idx1, idx2);
        assert_eq!(pool.len(), 1);
    }

    #[test]
    fn no_cross_type_dedup() {
        // f32 0.0 and f64 0.0 should NOT deduplicate (different sizes).
        let mut pool = ConstantPool::new();
        let idx_f32 = pool.add_f32(0.0);
        let idx_f64 = pool.add_f64(0.0);
        assert_ne!(idx_f32, idx_f64);
        assert_eq!(pool.len(), 2);
    }

    #[test]
    fn alignment_f32_then_f64() {
        let mut pool = ConstantPool::new();
        pool.add_f32(1.0); // 4 bytes at offset 0
        pool.add_f64(2.0); // 8 bytes, must align to 8 -> offset 8

        let total = pool.layout();
        assert_eq!(pool.entry_offset(0), 0);
        assert_eq!(pool.entry_offset(1), 8); // aligned to 8
        assert_eq!(total, 16); // 8 (aligned start) + 8 (data)

        let bytes = pool.emit();
        assert_eq!(bytes.len(), 16);
        // Check f32 at offset 0
        assert_eq!(&bytes[0..4], &1.0_f32.to_le_bytes());
        // Bytes 4..8 are padding
        assert_eq!(&bytes[4..8], &[0, 0, 0, 0]);
        // Check f64 at offset 8
        assert_eq!(&bytes[8..16], &2.0_f64.to_le_bytes());
    }

    #[test]
    fn alignment_f64_then_f32() {
        let mut pool = ConstantPool::new();
        pool.add_f64(3.0); // 8 bytes at offset 0
        pool.add_f32(4.0); // 4 bytes, align=4 -> offset 8

        let total = pool.layout();
        assert_eq!(pool.entry_offset(0), 0);
        assert_eq!(pool.entry_offset(1), 8);
        assert_eq!(total, 12);

        let bytes = pool.emit();
        assert_eq!(bytes.len(), 12);
        assert_eq!(&bytes[0..8], &3.0_f64.to_le_bytes());
        assert_eq!(&bytes[8..12], &4.0_f32.to_le_bytes());
    }

    #[test]
    fn multiple_f32_no_padding() {
        let mut pool = ConstantPool::new();
        pool.add_f32(1.0);
        pool.add_f32(2.0);
        pool.add_f32(3.0);

        let total = pool.layout();
        assert_eq!(pool.entry_offset(0), 0);
        assert_eq!(pool.entry_offset(1), 4);
        assert_eq!(pool.entry_offset(2), 8);
        assert_eq!(total, 12);

        let bytes = pool.emit();
        assert_eq!(bytes.len(), 12);
        assert_eq!(&bytes[0..4], &1.0_f32.to_le_bytes());
        assert_eq!(&bytes[4..8], &2.0_f32.to_le_bytes());
        assert_eq!(&bytes[8..12], &3.0_f32.to_le_bytes());
    }

    #[test]
    fn dedup_nan_f32() {
        // NaN has a specific bit pattern; two NaN values with same bits should dedup.
        let nan = f32::NAN;
        let mut pool = ConstantPool::new();
        let idx1 = pool.add_f32(nan);
        let idx2 = pool.add_f32(nan);
        assert_eq!(idx1, idx2);
        assert_eq!(pool.len(), 1);
    }

    #[test]
    fn dedup_nan_f64() {
        let nan = f64::NAN;
        let mut pool = ConstantPool::new();
        let idx1 = pool.add_f64(nan);
        let idx2 = pool.add_f64(nan);
        assert_eq!(idx1, idx2);
        assert_eq!(pool.len(), 1);
    }

    #[test]
    fn positive_and_negative_zero_f32() {
        // +0.0 and -0.0 have different bit patterns, should be separate entries.
        let mut pool = ConstantPool::new();
        let idx_pos = pool.add_f32(0.0_f32);
        let idx_neg = pool.add_f32(-0.0_f32);
        assert_ne!(idx_pos, idx_neg);
        assert_eq!(pool.len(), 2);
    }

    #[test]
    fn positive_and_negative_zero_f64() {
        let mut pool = ConstantPool::new();
        let idx_pos = pool.add_f64(0.0_f64);
        let idx_neg = pool.add_f64(-0.0_f64);
        assert_ne!(idx_pos, idx_neg);
        assert_eq!(pool.len(), 2);
    }

    #[test]
    fn large_pool_mixed_types() {
        let mut pool = ConstantPool::new();

        // Add a mix of f32 and f64 constants.
        let idx0 = pool.add_f32(1.0);
        let idx1 = pool.add_f64(2.0);
        let idx2 = pool.add_f32(3.0);
        let idx3 = pool.add_f64(4.0);
        let idx4 = pool.add_f32(1.0); // dedup with idx0

        assert_eq!(idx4, idx0);
        assert_eq!(pool.len(), 4);

        let total = pool.layout();
        // Entry 0: f32 at 0 (4 bytes)
        // Entry 1: f64 at 8 (aligned, 8 bytes)
        // Entry 2: f32 at 16 (4 bytes)
        // Entry 3: f64 at 24 (aligned, 8 bytes)
        assert_eq!(pool.entry_offset(idx0), 0);
        assert_eq!(pool.entry_offset(idx1), 8);
        assert_eq!(pool.entry_offset(idx2), 16);
        assert_eq!(pool.entry_offset(idx3), 24);
        assert_eq!(total, 32);

        let bytes = pool.emit();
        assert_eq!(bytes.len(), 32);
    }

    #[test]
    fn emit_before_layout_returns_zeros() {
        // Before layout, offsets are all 0, so emit piles everything at offset 0.
        // This is technically correct (just overlapping) — the important thing is
        // it doesn't panic.
        let mut pool = ConstantPool::new();
        pool.add_f32(1.0);
        pool.add_f32(2.0);
        let bytes = pool.emit();
        // Both entries at offset 0 -> last writer wins for overlapping bytes
        assert_eq!(bytes.len(), 4);
    }

    #[test]
    fn entries_accessor() {
        let mut pool = ConstantPool::new();
        pool.add_f32(42.0);
        pool.add_f64(84.0);
        let entries = pool.entries();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].align, 4);
        assert_eq!(entries[1].align, 8);
    }
}
