// llvm2-codegen/elf/debug.rs - DWARF debug sections and ELF metadata stubs
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Reference: DWARF Debugging Information Format Version 5 (2017-02-13)
// Reference: System V ABI, ELF-64 Object File Format (section groups, program headers)
// Reference: GNU Binutils (note.GNU-stack convention)

//! Minimal DWARF debug section stubs and ELF metadata support.
//!
//! This module provides:
//! - **DWARF debug stubs**: Minimal `.debug_info`, `.debug_abbrev`, `.debug_line`
//!   section content for DWARF v5 that tools like `readelf` can parse.
//! - **`.note.GNU-stack`**: Marks the object as not requiring an executable stack.
//! - **Section groups**: ELF COMDAT `.group` sections for deduplication.
//! - **Program header stubs**: `PT_LOAD` segment structures for future linking support.
//!
//! All helpers are self-contained and depend only on [`super::constants`].

#[allow(unused_imports)]
use super::constants::*;

// ---------------------------------------------------------------------------
// Additional ELF constants (not in constants.rs)
// ---------------------------------------------------------------------------

/// ELF section type for section groups.
pub const SHT_GROUP: u32 = 17;

/// ELF section type for notes.
pub const SHT_NOTE: u32 = 7;

/// Section flag: member of a section group.
pub const SHF_GROUP: u64 = 0x200;

/// COMDAT section group flag (first word of `.group` data).
pub const GRP_COMDAT: u32 = 1;

/// Loadable segment type.
pub const PT_LOAD: u32 = 1;

/// Unused program header entry type.
pub const PT_NULL: u32 = 0;

/// Segment permission: readable.
pub const PF_R: u32 = 4;

/// Segment permission: writable.
pub const PF_W: u32 = 2;

/// Segment permission: executable.
pub const PF_X: u32 = 1;

/// Size of an ELF64 program header entry in bytes.
pub const ELF64_PHDR_SIZE: usize = 56;

/// DWARF v5 compilation unit type tag.
pub const DW_UT_COMPILE: u8 = 0x01;

// ---------------------------------------------------------------------------
// Internal serialization helpers
// ---------------------------------------------------------------------------

fn push_u16(buf: &mut Vec<u8>, value: u16) {
    buf.extend_from_slice(&value.to_le_bytes());
}

fn push_u32(buf: &mut Vec<u8>, value: u32) {
    buf.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(buf: &mut Vec<u8>, value: u64) {
    buf.extend_from_slice(&value.to_le_bytes());
}

// ---------------------------------------------------------------------------
// DwarfDebugStubs
// ---------------------------------------------------------------------------

/// Minimal DWARF debug section stubs.
///
/// The emitted sections are intentionally tiny but structurally parseable:
/// - `.debug_info` contains only a DWARF v5 compilation unit header.
/// - `.debug_abbrev` contains a single null abbreviation terminator.
/// - `.debug_line` contains a DWARF v5 line table header with empty tables.
///
/// The `.debug_info` stub emits the minimal self-consistent `unit_length`
/// required for the v5 header body (8 bytes: version + unit_type +
/// address_size + debug_abbrev_offset).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DwarfDebugStubs {
    /// DWARF version to emit (default: 5).
    pub version: u16,
    /// Address size in bytes (default: 8 for 64-bit).
    pub address_size: u8,
    /// Unit type for `.debug_info` (default: DW_UT_COMPILE).
    pub unit_type: u8,
}

impl DwarfDebugStubs {
    /// Section name for `.debug_info`.
    pub const DEBUG_INFO_NAME: &'static str = ".debug_info";
    /// Section name for `.debug_abbrev`.
    pub const DEBUG_ABBREV_NAME: &'static str = ".debug_abbrev";
    /// Section name for `.debug_line`.
    pub const DEBUG_LINE_NAME: &'static str = ".debug_line";

    /// Create default DWARF v5 ELF64 debug stubs.
    pub fn new() -> Self {
        Self {
            version: 5,
            address_size: 8,
            unit_type: DW_UT_COMPILE,
        }
    }

    /// Generate `.debug_info` section bytes.
    ///
    /// Layout (DWARF v5, 32-bit DWARF format):
    /// ```text
    /// Offset  Size  Field
    /// 0        4    unit_length (excludes itself)
    /// 4        2    version (5)
    /// 6        1    unit_type (DW_UT_compile = 0x01)
    /// 7        1    address_size (8)
    /// 8        4    debug_abbrev_offset (0)
    /// ```
    pub fn debug_info_bytes(&self) -> Vec<u8> {
        // Payload: version(2) + unit_type(1) + address_size(1) + abbrev_offset(4) = 8 bytes
        let mut payload = Vec::with_capacity(8);
        push_u16(&mut payload, self.version);
        payload.push(self.unit_type);
        payload.push(self.address_size);
        push_u32(&mut payload, 0); // debug_abbrev_offset

        let mut bytes = Vec::with_capacity(4 + payload.len());
        push_u32(&mut bytes, payload.len() as u32); // unit_length
        bytes.extend_from_slice(&payload);
        bytes
    }

    /// Generate `.debug_abbrev` section bytes.
    ///
    /// A single null byte terminates the abbreviation table.
    pub fn debug_abbrev_bytes(&self) -> Vec<u8> {
        vec![0]
    }

    /// Generate `.debug_line` section bytes.
    ///
    /// Layout (DWARF v5, 32-bit DWARF format):
    /// ```text
    /// Offset  Size  Field
    /// 0        4    unit_length (excludes itself)
    /// 4        2    version (5)
    /// 6        1    address_size (8)
    /// 7        1    segment_selector_size (0)
    /// 8        4    header_length
    /// 12       1    minimum_instruction_length (1)
    /// 13       1    maximum_operations_per_instruction (1)
    /// 14       1    default_is_stmt (1)
    /// 15       1    line_base (0)
    /// 16       1    line_range (1)
    /// 17       1    opcode_base (1)
    /// 18       1    directory_entry_format_count (0)
    /// 19       1    directories_count (0)
    /// 20       1    file_name_entry_format_count (0)
    /// 21       1    file_names_count (0)
    /// ```
    pub fn debug_line_bytes(&self) -> Vec<u8> {
        // Header body after the header_length field
        let mut header_body = Vec::with_capacity(10);
        header_body.push(1); // minimum_instruction_length
        header_body.push(1); // maximum_operations_per_instruction
        header_body.push(1); // default_is_stmt
        header_body.push(0); // line_base
        header_body.push(1); // line_range
        header_body.push(1); // opcode_base (no standard opcode lengths)
        header_body.push(0); // directory_entry_format_count (ULEB128 0)
        header_body.push(0); // directories_count (ULEB128 0)
        header_body.push(0); // file_name_entry_format_count (ULEB128 0)
        header_body.push(0); // file_names_count (ULEB128 0)

        // Payload: version(2) + address_size(1) + segment_selector_size(1)
        //        + header_length(4) + header_body
        let mut payload = Vec::with_capacity(8 + header_body.len());
        push_u16(&mut payload, self.version);
        payload.push(self.address_size);
        payload.push(0); // segment_selector_size
        push_u32(&mut payload, header_body.len() as u32); // header_length
        payload.extend_from_slice(&header_body);

        let mut bytes = Vec::with_capacity(4 + payload.len());
        push_u32(&mut bytes, payload.len() as u32); // unit_length
        bytes.extend_from_slice(&payload);
        bytes
    }

    /// Generate all three DWARF stub sections as (name, bytes) tuples.
    pub fn sections(&self) -> [(&'static str, Vec<u8>); 3] {
        [
            (Self::DEBUG_INFO_NAME, self.debug_info_bytes()),
            (Self::DEBUG_ABBREV_NAME, self.debug_abbrev_bytes()),
            (Self::DEBUG_LINE_NAME, self.debug_line_bytes()),
        ]
    }
}

impl Default for DwarfDebugStubs {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// NoteGnuStack
// ---------------------------------------------------------------------------

/// `.note.GNU-stack` section marker.
///
/// An empty section of type `SHT_PROGBITS` with no `SHF_EXECINSTR` flag.
/// Its presence tells the linker that this object does not require an
/// executable stack. Without it, the GNU linker may assume the worst and
/// mark the stack executable.
///
/// Despite the ".note" prefix in the name, the conventional section type
/// is `SHT_PROGBITS` (not `SHT_NOTE`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NoteGnuStack;

impl NoteGnuStack {
    /// Section name.
    pub const NAME: &'static str = ".note.GNU-stack";
    /// Standard alignment for this section.
    pub const ALIGN: u64 = 1;

    /// Create a new `.note.GNU-stack` marker.
    pub fn new() -> Self {
        Self
    }

    /// ELF section type for this section.
    pub fn section_type(&self) -> u32 {
        SHT_PROGBITS
    }

    /// ELF section flags (no SHF_EXECINSTR = non-executable stack).
    pub fn section_flags(&self) -> u64 {
        0
    }

    /// Section alignment.
    pub fn section_align(&self) -> u64 {
        Self::ALIGN
    }

    /// Generate section contents (empty for this marker section).
    pub fn section_bytes(&self) -> Vec<u8> {
        Vec::new()
    }
}

// ---------------------------------------------------------------------------
// SectionGroup
// ---------------------------------------------------------------------------

/// ELF COMDAT section group (`.group` section).
///
/// A section group allows the linker to discard duplicate copies of a set of
/// sections across translation units. The section data is a sequence of
/// little-endian `u32` values: the group flag word (`GRP_COMDAT`) followed
/// by the section header indices of member sections.
///
/// The `.group` section header uses:
/// - `sh_type = SHT_GROUP`
/// - `sh_link = <.symtab section index>`
/// - `sh_info = <signature symbol index>`
/// - `sh_flags = 0` (the group section itself does NOT have SHF_GROUP;
///    member sections carry that flag)
/// - `sh_entsize = 4` (each entry is a u32)
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SectionGroup {
    /// Section header index of the associated symbol table (.symtab).
    pub symtab_section_index: u32,
    /// Symbol table index of the group signature symbol.
    pub signature_symbol_index: u32,
    /// Section header indices of the member sections.
    pub member_indices: Vec<u32>,
}

impl SectionGroup {
    /// Section name.
    pub const NAME: &'static str = ".group";
    /// Alignment for the group section.
    pub const ALIGN: u64 = 4;
    /// Entry size (each member index is a u32).
    pub const ENTRY_SIZE: u64 = 4;

    /// Create a new COMDAT section group.
    pub fn new(
        symtab_section_index: u32,
        signature_symbol_index: u32,
        member_indices: Vec<u32>,
    ) -> Self {
        Self {
            symtab_section_index,
            signature_symbol_index,
            member_indices,
        }
    }

    /// ELF section type for this section.
    pub fn section_type(&self) -> u32 {
        SHT_GROUP
    }

    /// ELF section flags.
    ///
    /// The `.group` section itself has flags = 0. Member sections carry
    /// `SHF_GROUP`.
    pub fn section_flags(&self) -> u64 {
        0
    }

    /// Section alignment.
    pub fn section_align(&self) -> u64 {
        Self::ALIGN
    }

    /// Group flag word (first u32 in the section data).
    pub fn group_flag(&self) -> u32 {
        GRP_COMDAT
    }

    /// Generate section contents: flag word followed by member section indices.
    pub fn section_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity((1 + self.member_indices.len()) * 4);
        push_u32(&mut bytes, self.group_flag());
        for &idx in &self.member_indices {
            push_u32(&mut bytes, idx);
        }
        bytes
    }
}

// ---------------------------------------------------------------------------
// ProgramHeaderStub
// ---------------------------------------------------------------------------

/// Minimal ELF64 program header entry for future linking support.
///
/// Program headers describe segments for the runtime loader. While
/// relocatable object files (.o) typically have no program headers,
/// this stub supports generating them for when LLVM2 produces
/// executable or shared library output.
///
/// Layout (Elf64_Phdr, 56 bytes):
/// ```text
/// Offset  Size  Field
/// 0        4    p_type
/// 4        4    p_flags
/// 8        8    p_offset
/// 16       8    p_vaddr
/// 24       8    p_paddr
/// 32       8    p_filesz
/// 40       8    p_memsz
/// 48       8    p_align
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProgramHeaderStub {
    /// Segment type (PT_LOAD, PT_NULL, etc.).
    pub p_type: u32,
    /// Segment permission flags (PF_R | PF_W | PF_X).
    pub p_flags: u32,
    /// File offset of the segment.
    pub p_offset: u64,
    /// Virtual address of the segment.
    pub p_vaddr: u64,
    /// Physical address of the segment.
    pub p_paddr: u64,
    /// Size of the segment in the file.
    pub p_filesz: u64,
    /// Size of the segment in memory.
    pub p_memsz: u64,
    /// Alignment (must be power of 2; 0 or 1 means no alignment).
    pub p_align: u64,
}

impl ProgramHeaderStub {
    /// Create a minimal read-only `PT_LOAD` program header.
    pub fn load() -> Self {
        Self {
            p_type: PT_LOAD,
            p_flags: PF_R,
            p_offset: 0,
            p_vaddr: 0,
            p_paddr: 0,
            p_filesz: 0,
            p_memsz: 0,
            p_align: 0,
        }
    }

    /// Create a null program header entry (PT_NULL, all zeros).
    pub fn null() -> Self {
        Self {
            p_type: PT_NULL,
            p_flags: 0,
            p_offset: 0,
            p_vaddr: 0,
            p_paddr: 0,
            p_filesz: 0,
            p_memsz: 0,
            p_align: 0,
        }
    }

    /// Size of the serialized program header entry in bytes.
    pub fn size() -> usize {
        ELF64_PHDR_SIZE
    }

    /// Serialize the program header to little-endian bytes (56 bytes).
    pub fn write(&self, buf: &mut Vec<u8>) {
        push_u32(buf, self.p_type);
        push_u32(buf, self.p_flags);
        push_u64(buf, self.p_offset);
        push_u64(buf, self.p_vaddr);
        push_u64(buf, self.p_paddr);
        push_u64(buf, self.p_filesz);
        push_u64(buf, self.p_memsz);
        push_u64(buf, self.p_align);
    }

    /// Serialize to a standalone byte vector.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(ELF64_PHDR_SIZE);
        self.write(&mut buf);
        buf
    }
}

impl Default for ProgramHeaderStub {
    fn default() -> Self {
        Self::load()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn read_u16_le(bytes: &[u8], offset: usize) -> u16 {
        u16::from_le_bytes([bytes[offset], bytes[offset + 1]])
    }

    fn read_u32_le(bytes: &[u8], offset: usize) -> u32 {
        u32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ])
    }

    fn read_u64_le(bytes: &[u8], offset: usize) -> u64 {
        u64::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
            bytes[offset + 4],
            bytes[offset + 5],
            bytes[offset + 6],
            bytes[offset + 7],
        ])
    }

    // --- Constants ---

    #[test]
    fn test_elf_constants_values() {
        assert_eq!(SHT_GROUP, 17);
        assert_eq!(SHT_NOTE, 7);
        assert_eq!(SHF_GROUP, 0x200);
        assert_eq!(GRP_COMDAT, 1);
        assert_eq!(PT_LOAD, 1);
        assert_eq!(PT_NULL, 0);
        assert_eq!(PF_R, 4);
        assert_eq!(PF_W, 2);
        assert_eq!(PF_X, 1);
        assert_eq!(ELF64_PHDR_SIZE, 56);
        assert_eq!(DW_UT_COMPILE, 0x01);
    }

    // --- DwarfDebugStubs ---

    #[test]
    fn test_dwarf_stubs_default() {
        let stubs = DwarfDebugStubs::new();
        assert_eq!(stubs.version, 5);
        assert_eq!(stubs.address_size, 8);
        assert_eq!(stubs.unit_type, DW_UT_COMPILE);
    }

    #[test]
    fn test_debug_info_layout() {
        let bytes = DwarfDebugStubs::new().debug_info_bytes();
        // Total: 4 (unit_length) + 8 (payload) = 12 bytes
        assert_eq!(bytes.len(), 12);
        // unit_length = 8 (excludes the 4-byte length field itself)
        assert_eq!(read_u32_le(&bytes, 0), 8);
        assert_eq!(read_u32_le(&bytes, 0) as usize + 4, bytes.len());
        // version = 5
        assert_eq!(read_u16_le(&bytes, 4), 5);
        // unit_type = DW_UT_compile
        assert_eq!(bytes[6], DW_UT_COMPILE);
        // address_size = 8
        assert_eq!(bytes[7], 8);
        // debug_abbrev_offset = 0
        assert_eq!(read_u32_le(&bytes, 8), 0);
    }

    #[test]
    fn test_debug_abbrev_layout() {
        let bytes = DwarfDebugStubs::new().debug_abbrev_bytes();
        assert_eq!(bytes, vec![0], "single null terminator");
    }

    #[test]
    fn test_debug_line_layout() {
        let bytes = DwarfDebugStubs::new().debug_line_bytes();
        // Total: 4 (unit_length) + 18 (payload) = 22 bytes
        assert_eq!(bytes.len(), 22);
        // unit_length = 18
        assert_eq!(read_u32_le(&bytes, 0), 18);
        assert_eq!(read_u32_le(&bytes, 0) as usize + 4, bytes.len());
        // version = 5
        assert_eq!(read_u16_le(&bytes, 4), 5);
        // address_size = 8
        assert_eq!(bytes[6], 8);
        // segment_selector_size = 0
        assert_eq!(bytes[7], 0);
        // header_length = 10
        assert_eq!(read_u32_le(&bytes, 8), 10);
        // minimum_instruction_length = 1
        assert_eq!(bytes[12], 1);
        // maximum_operations_per_instruction = 1
        assert_eq!(bytes[13], 1);
        // default_is_stmt = 1
        assert_eq!(bytes[14], 1);
        // line_base = 0
        assert_eq!(bytes[15], 0);
        // line_range = 1
        assert_eq!(bytes[16], 1);
        // opcode_base = 1
        assert_eq!(bytes[17], 1);
        // directory_entry_format_count = 0
        assert_eq!(bytes[18], 0);
        // directories_count = 0
        assert_eq!(bytes[19], 0);
        // file_name_entry_format_count = 0
        assert_eq!(bytes[20], 0);
        // file_names_count = 0
        assert_eq!(bytes[21], 0);
    }

    #[test]
    fn test_dwarf_sections_listing() {
        let stubs = DwarfDebugStubs::new();
        let sections = stubs.sections();
        assert_eq!(sections[0].0, ".debug_info");
        assert_eq!(sections[1].0, ".debug_abbrev");
        assert_eq!(sections[2].0, ".debug_line");
        assert_eq!(sections[0].1, stubs.debug_info_bytes());
        assert_eq!(sections[1].1, stubs.debug_abbrev_bytes());
        assert_eq!(sections[2].1, stubs.debug_line_bytes());
    }

    #[test]
    fn test_dwarf_section_names_constants() {
        assert_eq!(DwarfDebugStubs::DEBUG_INFO_NAME, ".debug_info");
        assert_eq!(DwarfDebugStubs::DEBUG_ABBREV_NAME, ".debug_abbrev");
        assert_eq!(DwarfDebugStubs::DEBUG_LINE_NAME, ".debug_line");
    }

    // --- NoteGnuStack ---

    #[test]
    fn test_note_gnu_stack_metadata() {
        let note = NoteGnuStack::new();
        assert_eq!(NoteGnuStack::NAME, ".note.GNU-stack");
        assert_eq!(note.section_type(), SHT_PROGBITS);
        assert_eq!(note.section_flags(), 0);
        assert_eq!(note.section_flags() & SHF_EXECINSTR, 0, "must not be executable");
        assert_eq!(note.section_align(), 1);
        assert!(note.section_bytes().is_empty(), "marker section has no data");
    }

    #[test]
    fn test_note_gnu_stack_default() {
        let note = NoteGnuStack;
        assert_eq!(note, NoteGnuStack::new());
    }

    // --- SectionGroup ---

    #[test]
    fn test_section_group_encoding() {
        let group = SectionGroup::new(6, 2, vec![1, 3, 7]);
        let bytes = group.section_bytes();

        assert_eq!(SectionGroup::NAME, ".group");
        assert_eq!(group.section_type(), SHT_GROUP);
        // Group section itself has flags=0 (members have SHF_GROUP)
        assert_eq!(group.section_flags(), 0);
        assert_eq!(group.section_align(), 4);
        assert_eq!(group.group_flag(), GRP_COMDAT);

        // 4 bytes flag + 3 * 4 bytes members = 16 bytes
        assert_eq!(bytes.len(), 16);
        assert_eq!(read_u32_le(&bytes, 0), GRP_COMDAT);
        assert_eq!(read_u32_le(&bytes, 4), 1);
        assert_eq!(read_u32_le(&bytes, 8), 3);
        assert_eq!(read_u32_le(&bytes, 12), 7);
    }

    #[test]
    fn test_section_group_empty() {
        let group = SectionGroup::new(4, 1, Vec::new());
        let bytes = group.section_bytes();
        // Still has the flag word
        assert_eq!(bytes.len(), 4);
        assert_eq!(read_u32_le(&bytes, 0), GRP_COMDAT);
    }

    #[test]
    fn test_section_group_fields() {
        let group = SectionGroup::new(5, 3, vec![2, 4]);
        assert_eq!(group.symtab_section_index, 5);
        assert_eq!(group.signature_symbol_index, 3);
        assert_eq!(group.member_indices, vec![2, 4]);
    }

    #[test]
    fn test_section_group_entry_size() {
        assert_eq!(SectionGroup::ENTRY_SIZE, 4);
    }

    // --- ProgramHeaderStub ---

    #[test]
    fn test_program_header_load_encoding() {
        let phdr = ProgramHeaderStub::load();
        let bytes = phdr.to_bytes();

        assert_eq!(ProgramHeaderStub::size(), ELF64_PHDR_SIZE);
        assert_eq!(bytes.len(), ELF64_PHDR_SIZE);
        assert_eq!(read_u32_le(&bytes, 0), PT_LOAD);
        assert_eq!(read_u32_le(&bytes, 4), PF_R);
        assert_eq!(read_u64_le(&bytes, 8), 0);  // p_offset
        assert_eq!(read_u64_le(&bytes, 16), 0); // p_vaddr
        assert_eq!(read_u64_le(&bytes, 24), 0); // p_paddr
        assert_eq!(read_u64_le(&bytes, 32), 0); // p_filesz
        assert_eq!(read_u64_le(&bytes, 40), 0); // p_memsz
        assert_eq!(read_u64_le(&bytes, 48), 0); // p_align
    }

    #[test]
    fn test_program_header_null() {
        let phdr = ProgramHeaderStub::null();
        let bytes = phdr.to_bytes();
        assert_eq!(bytes.len(), ELF64_PHDR_SIZE);
        assert_eq!(read_u32_le(&bytes, 0), PT_NULL);
        // All remaining fields should be zero
        assert!(bytes[4..].iter().all(|&b| b == 0), "null phdr should be zeros after type");
    }

    #[test]
    fn test_program_header_custom() {
        let phdr = ProgramHeaderStub {
            p_type: PT_LOAD,
            p_flags: PF_R | PF_X,
            p_offset: 0x40,
            p_vaddr: 0x400000,
            p_paddr: 0x400000,
            p_filesz: 0x120,
            p_memsz: 0x180,
            p_align: 0x1000,
        };
        let bytes = phdr.to_bytes();

        assert_eq!(read_u32_le(&bytes, 0), PT_LOAD);
        assert_eq!(read_u32_le(&bytes, 4), PF_R | PF_X);
        assert_eq!(read_u64_le(&bytes, 8), 0x40);
        assert_eq!(read_u64_le(&bytes, 16), 0x400000);
        assert_eq!(read_u64_le(&bytes, 24), 0x400000);
        assert_eq!(read_u64_le(&bytes, 32), 0x120);
        assert_eq!(read_u64_le(&bytes, 40), 0x180);
        assert_eq!(read_u64_le(&bytes, 48), 0x1000);
    }

    #[test]
    fn test_program_header_default_is_load() {
        let phdr = ProgramHeaderStub::default();
        assert_eq!(phdr, ProgramHeaderStub::load());
    }

    #[test]
    fn test_program_header_write_method() {
        let phdr = ProgramHeaderStub::load();
        let mut buf = Vec::new();
        phdr.write(&mut buf);
        assert_eq!(buf, phdr.to_bytes());
    }

    #[test]
    fn test_rwx_flag_combinations() {
        assert_eq!(PF_R | PF_W | PF_X, 7);
        assert_eq!(PF_R | PF_W, 6);
        assert_eq!(PF_R | PF_X, 5);
    }
}
