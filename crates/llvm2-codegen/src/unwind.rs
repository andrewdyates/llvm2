// llvm2-codegen - Compact unwind info emission for AArch64 macOS
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Reference: ~/llvm-project-ref/llvm/lib/MC/MCObjectFileInfo.cpp
//            (compact unwind section creation, line 213)
// Reference: ~/llvm-project-ref/llvm/lib/Target/AArch64/MCTargetDesc/AArch64AsmBackend.cpp
//            (generateCompactUnwindEncoding, line 576)
// Reference: Apple Mach-O compact_unwind_encoding.h

//! AArch64 Darwin compact unwind info emission.
//!
//! On macOS, the `__LD,__compact_unwind` section is required for usable
//! binaries. Each function gets a 32-byte entry that encodes how to unwind
//! its stack frame. The linker (`ld`) processes these entries and produces
//! the final `__TEXT,__unwind_info` section in the linked binary.
//!
//! # Compact unwind entry format
//!
//! ```text
//! struct compact_unwind_entry {
//!     uint64_t function_offset;     // start address of function (relocated)
//!     uint32_t function_length;     // length of function in bytes
//!     uint32_t compact_encoding;    // unwind encoding (from frame.rs)
//!     uint64_t personality;         // personality function ptr (0 for C)
//!     uint64_t lsda;               // language-specific data area (0 for C)
//! }
//! // Total: 32 bytes per entry, little-endian
//! ```
//!
//! # Relocations
//!
//! The `function_offset` field requires an `ARM64_RELOC_UNSIGNED` relocation
//! (length=3, quad) pointing to the function's symbol. Personality and LSDA
//! fields also need relocations when non-zero, but for C/Rust code without
//! exception handling they are zero.
//!
//! # Section attributes
//!
//! The section is `__LD,__compact_unwind` with `S_ATTR_DEBUG` flag. It lives
//! in the object file's single unnamed segment alongside `__text` and `__data`.

use crate::frame::{encode_compact_unwind, FrameLayout};
use crate::macho::constants::{ARM64_RELOC_UNSIGNED, RELOC_LENGTH_QUAD};
use crate::macho::writer::{MachOWriter, Relocation};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Size of one compact unwind entry in bytes.
pub const COMPACT_UNWIND_ENTRY_SIZE: u32 = 32;

/// Section attribute: debug information section.
/// Reference: mach-o/loader.h S_ATTR_DEBUG
pub const S_ATTR_DEBUG: u32 = 0x0200_0000;

// ---------------------------------------------------------------------------
// CompactUnwindEntry
// ---------------------------------------------------------------------------

/// A single compact unwind entry for one function.
///
/// Represents the 32-byte `compact_unwind_entry` structure that the
/// macOS linker uses to build the final `__TEXT,__unwind_info` section.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompactUnwindEntry {
    /// Virtual address of the function start (will be relocated).
    pub function_offset: u64,
    /// Length of the function in bytes.
    pub function_length: u32,
    /// Compact unwind encoding (from `encode_compact_unwind()`).
    pub compact_encoding: u32,
    /// Personality function pointer (0 for C/Rust without exceptions).
    pub personality: u64,
    /// Language-specific data area pointer (0 for C/Rust without exceptions).
    pub lsda: u64,
    /// Symbol index for the function (used to generate relocations).
    pub symbol_index: u32,
    /// Symbol index for the personality function (0 if personality is 0).
    pub personality_symbol_index: u32,
    /// Symbol index for the LSDA (0 if lsda is 0).
    pub lsda_symbol_index: u32,
}

impl CompactUnwindEntry {
    /// Create a new compact unwind entry for a simple C/Rust function
    /// (no personality, no LSDA).
    pub fn new(
        function_offset: u64,
        function_length: u32,
        compact_encoding: u32,
        symbol_index: u32,
    ) -> Self {
        Self {
            function_offset,
            function_length,
            compact_encoding,
            personality: 0,
            lsda: 0,
            symbol_index,
            personality_symbol_index: 0,
            lsda_symbol_index: 0,
        }
    }

    /// Create a compact unwind entry from a FrameLayout and function metadata.
    pub fn from_layout(
        layout: &FrameLayout,
        function_offset: u64,
        function_length: u32,
        symbol_index: u32,
    ) -> Self {
        let encoding = encode_compact_unwind(layout);
        Self::new(
            function_offset,
            function_length,
            encoding.encoding,
            symbol_index,
        )
    }

    /// Serialize this entry to 32 bytes (little-endian).
    pub fn to_bytes(&self) -> [u8; 32] {
        let mut buf = [0u8; 32];
        buf[0..8].copy_from_slice(&self.function_offset.to_le_bytes());
        buf[8..12].copy_from_slice(&self.function_length.to_le_bytes());
        buf[12..16].copy_from_slice(&self.compact_encoding.to_le_bytes());
        buf[16..24].copy_from_slice(&self.personality.to_le_bytes());
        buf[24..32].copy_from_slice(&self.lsda.to_le_bytes());
        buf
    }
}

// ---------------------------------------------------------------------------
// CompactUnwindReloc
// ---------------------------------------------------------------------------

/// A relocation needed by the compact unwind section.
///
/// Each function_offset field needs an ARM64_RELOC_UNSIGNED relocation
/// pointing to the function's symbol. Personality and LSDA fields also
/// need relocations when non-zero.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompactUnwindReloc {
    /// Byte offset within the compact unwind section data.
    pub offset: u32,
    /// Symbol index in the Mach-O symbol table.
    pub symbol_index: u32,
}

// ---------------------------------------------------------------------------
// CompactUnwindSection
// ---------------------------------------------------------------------------

/// Collects compact unwind entries and emits the `__LD,__compact_unwind`
/// section data and relocations.
///
/// # Usage
///
/// ```ignore
/// let mut section = CompactUnwindSection::new();
/// section.add_entry(CompactUnwindEntry::new(0, 64, encoding, 0));
/// let bytes = section.to_bytes();
/// let relocs = section.relocations();
/// ```
#[derive(Debug, Clone)]
pub struct CompactUnwindSection {
    entries: Vec<CompactUnwindEntry>,
}

impl CompactUnwindSection {
    /// Create a new empty compact unwind section.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Add a compact unwind entry for a function.
    pub fn add_entry(&mut self, entry: CompactUnwindEntry) {
        self.entries.push(entry);
    }

    /// Returns the number of entries in this section.
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Returns true if the section has no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Serialize all entries to bytes.
    ///
    /// Returns a byte vector of length `entry_count * 32`.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(self.entries.len() * COMPACT_UNWIND_ENTRY_SIZE as usize);
        for entry in &self.entries {
            buf.extend_from_slice(&entry.to_bytes());
        }
        buf
    }

    /// Generate the relocations needed for this section.
    ///
    /// Each entry generates at least one relocation for the `function_offset`
    /// field (ARM64_RELOC_UNSIGNED, quad). Personality and LSDA fields
    /// generate additional relocations when non-zero.
    pub fn relocations(&self) -> Vec<CompactUnwindReloc> {
        let mut relocs = Vec::new();
        for (i, entry) in self.entries.iter().enumerate() {
            let base_offset = (i as u32) * COMPACT_UNWIND_ENTRY_SIZE;

            // function_offset at byte 0 of each entry
            relocs.push(CompactUnwindReloc {
                offset: base_offset,
                symbol_index: entry.symbol_index,
            });

            // personality at byte 16 of each entry (if non-zero)
            if entry.personality != 0 {
                relocs.push(CompactUnwindReloc {
                    offset: base_offset + 16,
                    symbol_index: entry.personality_symbol_index,
                });
            }

            // lsda at byte 24 of each entry (if non-zero)
            if entry.lsda != 0 {
                relocs.push(CompactUnwindReloc {
                    offset: base_offset + 24,
                    symbol_index: entry.lsda_symbol_index,
                });
            }
        }
        relocs
    }

    /// Total size of the section data in bytes.
    pub fn data_size(&self) -> u32 {
        (self.entries.len() as u32) * COMPACT_UNWIND_ENTRY_SIZE
    }
}

impl Default for CompactUnwindSection {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// MachOWriter integration
// ---------------------------------------------------------------------------

/// Add the compact unwind section to a MachOWriter.
///
/// This adds the `__LD,__compact_unwind` section with the serialized
/// entry data and the required ARM64_RELOC_UNSIGNED relocations for
/// each function_offset field.
///
/// # Arguments
/// * `writer` - The MachOWriter to add the section to.
/// * `section` - The compact unwind section with all entries.
///
/// # Returns
/// The 0-based section index of the newly added compact unwind section,
/// or `None` if the section is empty (no entries to emit).
pub fn add_compact_unwind_to_writer(
    writer: &mut MachOWriter,
    section: &CompactUnwindSection,
) -> Option<usize> {
    if section.is_empty() {
        return None;
    }

    let data = section.to_bytes();
    let section_index = writer.add_custom_section(
        b"__compact_unwind",
        b"__LD",
        &data,
        3, // 2^3 = 8-byte alignment
        S_ATTR_DEBUG,
    );

    // Add relocations for function_offset (and personality/lsda if needed).
    let relocs = section.relocations();
    for reloc in &relocs {
        writer.add_relocation(
            section_index,
            Relocation {
                offset: reloc.offset,
                symbol_index: reloc.symbol_index,
                pcrel: false,
                length: RELOC_LENGTH_QUAD,
                is_extern: true,
                reloc_type: ARM64_RELOC_UNSIGNED,
            },
        );
    }

    Some(section_index)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::{
        UNWIND_ARM64_FRAME_D8_D9_PAIR, UNWIND_ARM64_FRAME_X19_X20_PAIR, UNWIND_ARM64_MODE_DWARF,
        UNWIND_ARM64_MODE_FRAME, UNWIND_ARM64_MODE_FRAMELESS,
    };

    #[test]
    fn test_entry_size_is_32_bytes() {
        let entry = CompactUnwindEntry::new(0, 0, 0, 0);
        let bytes = entry.to_bytes();
        assert_eq!(bytes.len(), 32);
        assert_eq!(COMPACT_UNWIND_ENTRY_SIZE, 32);
    }

    #[test]
    fn test_entry_serialization_zeros() {
        let entry = CompactUnwindEntry::new(0, 0, 0, 0);
        let bytes = entry.to_bytes();
        assert_eq!(bytes, [0u8; 32]);
    }

    #[test]
    fn test_entry_serialization_fields() {
        let entry = CompactUnwindEntry::new(
            0x1000,                  // function_offset
            64,                      // function_length
            UNWIND_ARM64_MODE_FRAME, // compact_encoding
            0,                       // symbol_index (for relocs, not serialized)
        );
        let bytes = entry.to_bytes();

        // function_offset at bytes 0..8 (little-endian)
        assert_eq!(u64::from_le_bytes(bytes[0..8].try_into().unwrap()), 0x1000);
        // function_length at bytes 8..12
        assert_eq!(u32::from_le_bytes(bytes[8..12].try_into().unwrap()), 64);
        // compact_encoding at bytes 12..16
        assert_eq!(
            u32::from_le_bytes(bytes[12..16].try_into().unwrap()),
            UNWIND_ARM64_MODE_FRAME
        );
        // personality at bytes 16..24
        assert_eq!(u64::from_le_bytes(bytes[16..24].try_into().unwrap()), 0);
        // lsda at bytes 24..32
        assert_eq!(u64::from_le_bytes(bytes[24..32].try_into().unwrap()), 0);
    }

    #[test]
    fn test_entry_with_all_callee_saved() {
        let encoding = UNWIND_ARM64_MODE_FRAME
            | UNWIND_ARM64_FRAME_X19_X20_PAIR
            | UNWIND_ARM64_FRAME_D8_D9_PAIR;
        let entry = CompactUnwindEntry::new(0x2000, 128, encoding, 1);
        let bytes = entry.to_bytes();

        assert_eq!(
            u32::from_le_bytes(bytes[12..16].try_into().unwrap()),
            encoding
        );
        assert_eq!(u32::from_le_bytes(bytes[8..12].try_into().unwrap()), 128);
    }

    #[test]
    fn test_entry_with_personality() {
        let mut entry = CompactUnwindEntry::new(0x1000, 32, UNWIND_ARM64_MODE_FRAME, 0);
        entry.personality = 0xDEAD_BEEF;
        entry.personality_symbol_index = 5;
        let bytes = entry.to_bytes();

        assert_eq!(
            u64::from_le_bytes(bytes[16..24].try_into().unwrap()),
            0xDEAD_BEEF
        );
    }

    #[test]
    fn test_empty_section() {
        let section = CompactUnwindSection::new();
        assert_eq!(section.entry_count(), 0);
        assert!(section.is_empty());
        assert_eq!(section.to_bytes().len(), 0);
        assert_eq!(section.relocations().len(), 0);
        assert_eq!(section.data_size(), 0);
    }

    #[test]
    fn test_section_single_entry() {
        let mut section = CompactUnwindSection::new();
        section.add_entry(CompactUnwindEntry::new(
            0x1000,
            64,
            UNWIND_ARM64_MODE_FRAME,
            0,
        ));

        assert_eq!(section.entry_count(), 1);
        assert!(!section.is_empty());
        assert_eq!(section.data_size(), 32);

        let bytes = section.to_bytes();
        assert_eq!(bytes.len(), 32);

        // Verify the serialized data
        assert_eq!(u64::from_le_bytes(bytes[0..8].try_into().unwrap()), 0x1000);
    }

    #[test]
    fn test_section_multiple_entries() {
        let mut section = CompactUnwindSection::new();
        section.add_entry(CompactUnwindEntry::new(
            0x0000,
            32,
            UNWIND_ARM64_MODE_FRAME,
            0,
        ));
        section.add_entry(CompactUnwindEntry::new(
            0x0020,
            16,
            UNWIND_ARM64_MODE_FRAMELESS,
            1,
        ));
        section.add_entry(CompactUnwindEntry::new(
            0x0030,
            48,
            UNWIND_ARM64_MODE_DWARF,
            2,
        ));

        assert_eq!(section.entry_count(), 3);
        assert_eq!(section.data_size(), 96);

        let bytes = section.to_bytes();
        assert_eq!(bytes.len(), 96);

        // Verify each entry's function_offset
        assert_eq!(u64::from_le_bytes(bytes[0..8].try_into().unwrap()), 0x0000);
        assert_eq!(
            u64::from_le_bytes(bytes[32..40].try_into().unwrap()),
            0x0020
        );
        assert_eq!(
            u64::from_le_bytes(bytes[64..72].try_into().unwrap()),
            0x0030
        );

        // Verify each entry's encoding
        assert_eq!(
            u32::from_le_bytes(bytes[12..16].try_into().unwrap()),
            UNWIND_ARM64_MODE_FRAME
        );
        assert_eq!(
            u32::from_le_bytes(bytes[44..48].try_into().unwrap()),
            UNWIND_ARM64_MODE_FRAMELESS
        );
        assert_eq!(
            u32::from_le_bytes(bytes[76..80].try_into().unwrap()),
            UNWIND_ARM64_MODE_DWARF
        );
    }

    #[test]
    fn test_relocations_simple() {
        let mut section = CompactUnwindSection::new();
        section.add_entry(CompactUnwindEntry::new(0, 32, UNWIND_ARM64_MODE_FRAME, 0));
        section.add_entry(CompactUnwindEntry::new(0, 64, UNWIND_ARM64_MODE_FRAME, 1));

        let relocs = section.relocations();
        // Two entries, each with one relocation for function_offset
        assert_eq!(relocs.len(), 2);

        // First entry relocation at offset 0
        assert_eq!(relocs[0].offset, 0);
        assert_eq!(relocs[0].symbol_index, 0);

        // Second entry relocation at offset 32 (start of second entry)
        assert_eq!(relocs[1].offset, 32);
        assert_eq!(relocs[1].symbol_index, 1);
    }

    #[test]
    fn test_relocations_with_personality() {
        let mut section = CompactUnwindSection::new();

        let mut entry = CompactUnwindEntry::new(0, 32, UNWIND_ARM64_MODE_FRAME, 0);
        entry.personality = 0x1000; // non-zero triggers relocation
        entry.personality_symbol_index = 5;
        section.add_entry(entry);

        let relocs = section.relocations();
        // One function_offset reloc + one personality reloc
        assert_eq!(relocs.len(), 2);

        assert_eq!(relocs[0].offset, 0); // function_offset
        assert_eq!(relocs[0].symbol_index, 0);

        assert_eq!(relocs[1].offset, 16); // personality
        assert_eq!(relocs[1].symbol_index, 5);
    }

    #[test]
    fn test_relocations_with_lsda() {
        let mut section = CompactUnwindSection::new();

        let mut entry = CompactUnwindEntry::new(0, 32, UNWIND_ARM64_MODE_FRAME, 0);
        entry.lsda = 0x2000;
        entry.lsda_symbol_index = 10;
        section.add_entry(entry);

        let relocs = section.relocations();
        assert_eq!(relocs.len(), 2);

        assert_eq!(relocs[0].offset, 0); // function_offset
        assert_eq!(relocs[1].offset, 24); // lsda
        assert_eq!(relocs[1].symbol_index, 10);
    }

    #[test]
    fn test_relocations_with_all_fields() {
        let mut section = CompactUnwindSection::new();

        let mut entry = CompactUnwindEntry::new(0, 32, UNWIND_ARM64_MODE_FRAME, 0);
        entry.personality = 0x1000;
        entry.personality_symbol_index = 5;
        entry.lsda = 0x2000;
        entry.lsda_symbol_index = 10;
        section.add_entry(entry);

        let relocs = section.relocations();
        // function_offset + personality + lsda = 3 relocs
        assert_eq!(relocs.len(), 3);

        assert_eq!(relocs[0].offset, 0);
        assert_eq!(relocs[1].offset, 16);
        assert_eq!(relocs[2].offset, 24);
    }

    #[test]
    fn test_relocation_offsets_multi_entry() {
        let mut section = CompactUnwindSection::new();
        for i in 0..5 {
            section.add_entry(CompactUnwindEntry::new(0, 32, UNWIND_ARM64_MODE_FRAME, i));
        }

        let relocs = section.relocations();
        assert_eq!(relocs.len(), 5);

        for (i, reloc) in relocs.iter().enumerate() {
            assert_eq!(reloc.offset, (i as u32) * COMPACT_UNWIND_ENTRY_SIZE);
            assert_eq!(reloc.symbol_index, i as u32);
        }
    }

    #[test]
    fn test_entry_round_trip() {
        // Verify exact byte layout matches the compact_unwind_entry struct.
        let entry = CompactUnwindEntry {
            function_offset: 0x0000_0000_0000_4000,
            function_length: 0x100,
            compact_encoding: 0x0400_0003, // FRAME + X19/X20 + X21/X22
            personality: 0,
            lsda: 0,
            symbol_index: 0,
            personality_symbol_index: 0,
            lsda_symbol_index: 0,
        };
        let bytes = entry.to_bytes();

        // Read back each field
        let fn_off = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
        let fn_len = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        let enc = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
        let pers = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
        let lsda_val = u64::from_le_bytes(bytes[24..32].try_into().unwrap());

        assert_eq!(fn_off, 0x4000);
        assert_eq!(fn_len, 0x100);
        assert_eq!(enc, 0x0400_0003);
        assert_eq!(pers, 0);
        assert_eq!(lsda_val, 0);
    }

    #[test]
    fn test_s_attr_debug_constant() {
        // Verify S_ATTR_DEBUG matches the Mach-O spec value.
        assert_eq!(S_ATTR_DEBUG, 0x0200_0000);
    }

    #[test]
    fn test_default_trait() {
        let section = CompactUnwindSection::default();
        assert!(section.is_empty());
    }
}
