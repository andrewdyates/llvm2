// llvm2-codegen/macho/symbol.rs - Mach-O symbol table and string table emission
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Reference: LLVM MachObjectWriter.cpp writeNlist(), <mach-o/nlist.h>
// nlist_64 struct layout (16 bytes):
//   n_strx:  u32  — string table index
//   n_type:  u8   — type field (N_UNDF=0x0, N_ABS=0x2, N_SECT=0xE, N_EXT=0x01, N_PEXT=0x10)
//   n_sect:  u8   — section number (1-based, 0 = NO_SECT)
//   n_desc:  u16  — description field
//   n_value: u64  — symbol value (address or offset)

//! Mach-O symbol table (nlist_64) and string table emission.
//!
//! The symbol table is written as LC_SYMTAB data: a contiguous array of
//! `nlist_64` entries (16 bytes each) followed by the string table. Symbols
//! are partitioned into local, external (defined + exported), and undefined
//! groups for LC_DYSYMTAB.

/// Mach-O symbol type constants from `<mach-o/nlist.h>`.
pub mod nlist_type {
    /// Undefined symbol (imported, not defined in this object).
    pub const N_UNDF: u8 = 0x00;
    /// Absolute symbol (not section-relative).
    pub const N_ABS: u8 = 0x02;
    /// Defined in a section (the normal case for code/data symbols).
    pub const N_SECT: u8 = 0x0E;
    /// Private external — visible to the static linker but not exported to dyld.
    pub const N_PEXT: u8 = 0x10;
    /// External symbol — visible outside this object file.
    pub const N_EXT: u8 = 0x01;
    /// Mask for the type bits (bits 1-3 of n_type, after masking with N_TYPE).
    pub const N_TYPE: u8 = 0x0E;
    /// Indirect symbol.
    pub const N_INDR: u8 = 0x0A;
}

/// Size of an nlist_64 entry in bytes.
pub const NLIST_64_SIZE: usize = 16;

/// A single nlist_64 symbol table entry.
///
/// This is the in-memory representation of the Mach-O symbol table entry.
/// The `typ` field combines the type bits (N_SECT, N_UNDF, N_ABS) with
/// the external (N_EXT) and private external (N_PEXT) flags.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NList64 {
    /// Index into the string table for this symbol's name.
    pub strx: u32,

    /// Type field: `N_TYPE` bits OR'd with `N_EXT` and/or `N_PEXT`.
    /// For a normal defined external symbol: `N_SECT | N_EXT` = `0x0F`.
    pub typ: u8,

    /// Section number (1-based). 0 (`NO_SECT`) for undefined/absolute symbols.
    pub sect: u8,

    /// Description field. Used for various purposes:
    /// - `REFERENCE_FLAG_*` for undefined symbols
    /// - `N_WEAK_DEF` (0x0080) for weak definitions
    /// - `N_NO_DEAD_STRIP` (0x0020) to prevent dead stripping
    pub desc: u16,

    /// Symbol value. For defined symbols, this is the address/offset within
    /// the section. For common symbols, this is the size.
    pub value: u64,
}

impl NList64 {
    /// Encode this nlist_64 entry to its 16-byte little-endian binary representation.
    pub fn encode(&self) -> [u8; NLIST_64_SIZE] {
        let mut buf = [0u8; NLIST_64_SIZE];
        buf[0..4].copy_from_slice(&self.strx.to_le_bytes());
        buf[4] = self.typ;
        buf[5] = self.sect;
        buf[6..8].copy_from_slice(&self.desc.to_le_bytes());
        buf[8..16].copy_from_slice(&self.value.to_le_bytes());
        buf
    }

    /// Decode an nlist_64 entry from its 16-byte little-endian binary representation.
    pub fn decode(bytes: &[u8; NLIST_64_SIZE]) -> Self {
        Self {
            strx: u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
            typ: bytes[4],
            sect: bytes[5],
            desc: u16::from_le_bytes([bytes[6], bytes[7]]),
            value: u64::from_le_bytes([
                bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14],
                bytes[15],
            ]),
        }
    }

    /// Returns true if this symbol is external (exported).
    pub fn is_external(&self) -> bool {
        self.typ & nlist_type::N_EXT != 0
    }

    /// Returns true if this symbol is private external.
    pub fn is_private_extern(&self) -> bool {
        self.typ & nlist_type::N_PEXT != 0
    }

    /// Returns true if this symbol is undefined (imported).
    pub fn is_undefined(&self) -> bool {
        (self.typ & nlist_type::N_TYPE) == nlist_type::N_UNDF
    }

    /// Returns true if this symbol is defined in a section.
    pub fn is_defined(&self) -> bool {
        (self.typ & nlist_type::N_TYPE) == nlist_type::N_SECT
    }
}

/// Classification of a symbol for LC_DYSYMTAB partitioning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SymbolCategory {
    /// Local symbol (not external, not undefined).
    Local,
    /// External defined symbol (N_EXT set, defined in a section).
    External,
    /// Undefined symbol (imported from another object/dylib).
    Undefined,
}

/// A Mach-O symbol table with integrated string table.
///
/// Symbols are maintained in three partitions for LC_DYSYMTAB:
/// 1. Local symbols (file-scope, not exported)
/// 2. External symbols (defined and exported)
/// 3. Undefined symbols (imported references)
///
/// The string table starts with a NUL byte (index 0 is the empty string),
/// per Mach-O convention.
#[derive(Debug, Clone)]
pub struct SymbolTable {
    /// Local symbols (not external, not undefined).
    local_symbols: Vec<NList64>,
    /// External defined symbols.
    external_symbols: Vec<NList64>,
    /// Undefined (imported) symbols.
    undefined_symbols: Vec<NList64>,
    /// String table bytes. Starts with \0 (empty string at index 0).
    string_table: Vec<u8>,
}

impl SymbolTable {
    /// Create a new empty symbol table.
    ///
    /// The string table is initialized with a leading NUL byte, so index 0
    /// refers to the empty string.
    pub fn new() -> Self {
        Self {
            local_symbols: Vec::new(),
            external_symbols: Vec::new(),
            undefined_symbols: Vec::new(),
            string_table: vec![0], // Leading NUL byte
        }
    }

    /// Add a defined symbol in a section.
    ///
    /// # Arguments
    /// - `name`: Symbol name (will be added to string table)
    /// - `section`: 1-based section number
    /// - `value`: Address/offset within section
    /// - `is_global`: If true, symbol is external (N_EXT)
    /// - `is_private_extern`: If true, symbol is private external (N_PEXT)
    ///
    /// # Returns
    /// The symbol's index in the final (concatenated) symbol table.
    pub fn add_symbol(
        &mut self,
        name: &str,
        section: u8,
        value: u64,
        is_global: bool,
        is_private_extern: bool,
    ) -> u32 {
        let strx = self.add_string(name);

        let mut typ = nlist_type::N_SECT;
        if is_global {
            typ |= nlist_type::N_EXT;
        }
        if is_private_extern {
            typ |= nlist_type::N_PEXT;
        }

        let nlist = NList64 {
            strx,
            typ,
            sect: section,
            desc: 0,
            value,
        };

        if is_global || is_private_extern {
            let idx =
                self.local_symbols.len() + self.external_symbols.len();
            self.external_symbols.push(nlist);
            idx as u32
        } else {
            let idx = self.local_symbols.len();
            self.local_symbols.push(nlist);
            idx as u32
        }
    }

    /// Add an undefined (imported) symbol.
    ///
    /// # Returns
    /// The symbol's index in the final (concatenated) symbol table.
    pub fn add_undefined_symbol(&mut self, name: &str) -> u32 {
        let strx = self.add_string(name);

        let nlist = NList64 {
            strx,
            typ: nlist_type::N_UNDF | nlist_type::N_EXT,
            sect: 0, // NO_SECT
            desc: 0,
            value: 0,
        };

        let idx = self.local_symbols.len()
            + self.external_symbols.len()
            + self.undefined_symbols.len();
        self.undefined_symbols.push(nlist);
        idx as u32
    }

    /// Add a string to the string table and return its index.
    ///
    /// The string is NUL-terminated in the table.
    fn add_string(&mut self, s: &str) -> u32 {
        // Check if string already exists (simple dedup for common case)
        // The string table format is: \0name1\0name2\0...
        // We search for existing entries to avoid duplicates.
        if let Some(idx) = self.find_string(s) {
            return idx;
        }

        let idx = self.string_table.len() as u32;
        self.string_table.extend_from_slice(s.as_bytes());
        self.string_table.push(0); // NUL terminator
        idx
    }

    /// Find an existing string in the string table.
    fn find_string(&self, s: &str) -> Option<u32> {
        if s.is_empty() {
            return Some(0);
        }

        let needle = s.as_bytes();
        let table = &self.string_table;

        // Walk the string table looking for exact matches
        let mut pos = 1; // Skip leading NUL
        while pos < table.len() {
            let end = table[pos..]
                .iter()
                .position(|&b| b == 0)
                .map(|i| pos + i)
                .unwrap_or(table.len());

            if &table[pos..end] == needle {
                return Some(pos as u32);
            }

            pos = end + 1; // Skip NUL terminator
        }

        None
    }

    /// Total number of symbols across all categories.
    pub fn num_symbols(&self) -> u32 {
        (self.local_symbols.len() + self.external_symbols.len() + self.undefined_symbols.len())
            as u32
    }

    /// Number of local symbols.
    pub fn num_local_symbols(&self) -> u32 {
        self.local_symbols.len() as u32
    }

    /// Number of external defined symbols.
    pub fn num_external_symbols(&self) -> u32 {
        self.external_symbols.len() as u32
    }

    /// Number of undefined symbols.
    pub fn num_undefined_symbols(&self) -> u32 {
        self.undefined_symbols.len() as u32
    }

    /// Index of the first local symbol (always 0).
    pub fn local_symbols_index(&self) -> u32 {
        0
    }

    /// Index of the first external symbol.
    pub fn external_symbols_index(&self) -> u32 {
        self.local_symbols.len() as u32
    }

    /// Index of the first undefined symbol.
    pub fn undefined_symbols_index(&self) -> u32 {
        (self.local_symbols.len() + self.external_symbols.len()) as u32
    }

    /// Write the symbol table (all nlist_64 entries) as a contiguous byte vector.
    ///
    /// Symbols are written in partition order: local, external, undefined.
    /// Each entry is 16 bytes.
    pub fn write_symtab(&self) -> Vec<u8> {
        let total_entries = self.num_symbols() as usize;
        let mut buf = Vec::with_capacity(total_entries * NLIST_64_SIZE);

        for sym in self
            .local_symbols
            .iter()
            .chain(self.external_symbols.iter())
            .chain(self.undefined_symbols.iter())
        {
            buf.extend_from_slice(&sym.encode());
        }

        buf
    }

    /// Write the string table as a byte vector.
    ///
    /// The string table starts with a NUL byte and contains NUL-terminated strings.
    /// Per Mach-O convention, the size should be rounded up to a multiple of the
    /// pointer size, but that alignment is left to the object file writer.
    pub fn write_strtab(&self) -> Vec<u8> {
        self.string_table.clone()
    }

    /// Get the string table size in bytes.
    pub fn strtab_size(&self) -> u32 {
        self.string_table.len() as u32
    }

    /// Get a reference to a symbol by its index in the concatenated table.
    pub fn get_symbol(&self, index: u32) -> Option<&NList64> {
        let idx = index as usize;
        let local_len = self.local_symbols.len();
        let external_len = self.external_symbols.len();

        if idx < local_len {
            Some(&self.local_symbols[idx])
        } else if idx < local_len + external_len {
            Some(&self.external_symbols[idx - local_len])
        } else {
            let undef_idx = idx - local_len - external_len;
            self.undefined_symbols.get(undef_idx)
        }
    }

    /// Return the category of a symbol at the given index.
    pub fn symbol_category(&self, index: u32) -> Option<SymbolCategory> {
        let idx = index as usize;
        let local_len = self.local_symbols.len();
        let external_len = self.external_symbols.len();
        let undefined_len = self.undefined_symbols.len();

        if idx < local_len {
            Some(SymbolCategory::Local)
        } else if idx < local_len + external_len {
            Some(SymbolCategory::External)
        } else if idx < local_len + external_len + undefined_len {
            Some(SymbolCategory::Undefined)
        } else {
            None
        }
    }

    /// Generate LC_DYSYMTAB parameters.
    ///
    /// Returns `(ilocalsym, nlocalsym, iextdefsym, nextdefsym, iundefsym, nundefsym)`:
    /// the starting index and count for each symbol partition.
    pub fn dysymtab_params(&self) -> DysymtabParams {
        DysymtabParams {
            ilocalsym: self.local_symbols_index(),
            nlocalsym: self.num_local_symbols(),
            iextdefsym: self.external_symbols_index(),
            nextdefsym: self.num_external_symbols(),
            iundefsym: self.undefined_symbols_index(),
            nundefsym: self.num_undefined_symbols(),
        }
    }
}

impl Default for SymbolTable {
    fn default() -> Self {
        Self::new()
    }
}

/// Parameters for the LC_DYSYMTAB load command.
///
/// Describes the partitioning of the symbol table into local, external
/// defined, and undefined symbol ranges.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DysymtabParams {
    /// Index of the first local symbol.
    pub ilocalsym: u32,
    /// Number of local symbols.
    pub nlocalsym: u32,
    /// Index of the first external defined symbol.
    pub iextdefsym: u32,
    /// Number of external defined symbols.
    pub nextdefsym: u32,
    /// Index of the first undefined symbol.
    pub iundefsym: u32,
    /// Number of undefined symbols.
    pub nundefsym: u32,
}

impl DysymtabParams {
    /// Encode the LC_DYSYMTAB load command (80 bytes).
    ///
    /// This writes a complete `dysymtab_command` struct. Fields not related
    /// to symbol partitioning (TOC, module table, external relocs, indirect
    /// symbols, local relocs) are set to zero. The caller should fill those
    /// in if needed for dynamic linking.
    pub fn encode_load_command(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(80);

        // cmd: LC_DYSYMTAB = 0x0B
        buf.extend_from_slice(&0x0B_u32.to_le_bytes());
        // cmdsize: 80
        buf.extend_from_slice(&80_u32.to_le_bytes());

        // Symbol partition fields
        buf.extend_from_slice(&self.ilocalsym.to_le_bytes());
        buf.extend_from_slice(&self.nlocalsym.to_le_bytes());
        buf.extend_from_slice(&self.iextdefsym.to_le_bytes());
        buf.extend_from_slice(&self.nextdefsym.to_le_bytes());
        buf.extend_from_slice(&self.iundefsym.to_le_bytes());
        buf.extend_from_slice(&self.nundefsym.to_le_bytes());

        // tocoff, ntoc (table of contents — unused for .o files)
        buf.extend_from_slice(&0_u32.to_le_bytes());
        buf.extend_from_slice(&0_u32.to_le_bytes());

        // modtaboff, nmodtab (module table — unused for .o files)
        buf.extend_from_slice(&0_u32.to_le_bytes());
        buf.extend_from_slice(&0_u32.to_le_bytes());

        // extrefsymoff, nextrefsyms (external reference table)
        buf.extend_from_slice(&0_u32.to_le_bytes());
        buf.extend_from_slice(&0_u32.to_le_bytes());

        // indirectsymoff, nindirectsyms (indirect symbol table)
        buf.extend_from_slice(&0_u32.to_le_bytes());
        buf.extend_from_slice(&0_u32.to_le_bytes());

        // extreloff, nextrel (external relocations)
        buf.extend_from_slice(&0_u32.to_le_bytes());
        buf.extend_from_slice(&0_u32.to_le_bytes());

        // locreloff, nlocrel (local relocations)
        buf.extend_from_slice(&0_u32.to_le_bytes());
        buf.extend_from_slice(&0_u32.to_le_bytes());

        debug_assert_eq!(buf.len(), 80);
        buf
    }
}

/// Encode an LC_BUILD_VERSION load command for macOS arm64.
///
/// This load command specifies the platform, minimum OS version, and SDK version.
/// For Apple Silicon macOS, the platform is `PLATFORM_MACOS` (1).
///
/// # Arguments
/// - `minos`: Minimum OS version encoded as `(major << 16) | (minor << 8) | patch`.
///   For example, macOS 14.0.0 = `0x000E_0000`.
/// - `sdk`: SDK version, same encoding. For example, macOS 14.0 = `0x000E_0000`.
///
/// The command includes space for tool entries (ntools), but we emit zero tools
/// since this is a compiler-generated object file.
pub fn encode_build_version(minos: u32, sdk: u32) -> Vec<u8> {
    let mut buf = Vec::with_capacity(24);

    // cmd: LC_BUILD_VERSION = 0x32
    buf.extend_from_slice(&0x32_u32.to_le_bytes());
    // cmdsize: 24 (no tool entries)
    buf.extend_from_slice(&24_u32.to_le_bytes());
    // platform: PLATFORM_MACOS = 1
    buf.extend_from_slice(&1_u32.to_le_bytes());
    // minos
    buf.extend_from_slice(&minos.to_le_bytes());
    // sdk
    buf.extend_from_slice(&sdk.to_le_bytes());
    // ntools: 0
    buf.extend_from_slice(&0_u32.to_le_bytes());

    debug_assert_eq!(buf.len(), 24);
    buf
}

/// Encode an LC_BUILD_VERSION with a tool entry (e.g., ld version).
///
/// # Arguments
/// - `minos`: Minimum OS version.
/// - `sdk`: SDK version.
/// - `tool`: Tool type (3 = ld, 1 = clang).
/// - `tool_version`: Tool version encoded as `(major << 16) | (minor << 8) | patch`.
pub fn encode_build_version_with_tool(
    minos: u32,
    sdk: u32,
    tool: u32,
    tool_version: u32,
) -> Vec<u8> {
    let mut buf = Vec::with_capacity(32);

    // cmd: LC_BUILD_VERSION = 0x32
    buf.extend_from_slice(&0x32_u32.to_le_bytes());
    // cmdsize: 32 (one tool entry = 24 + 8)
    buf.extend_from_slice(&32_u32.to_le_bytes());
    // platform: PLATFORM_MACOS = 1
    buf.extend_from_slice(&1_u32.to_le_bytes());
    // minos
    buf.extend_from_slice(&minos.to_le_bytes());
    // sdk
    buf.extend_from_slice(&sdk.to_le_bytes());
    // ntools: 1
    buf.extend_from_slice(&1_u32.to_le_bytes());

    // build_tool_version entry
    buf.extend_from_slice(&tool.to_le_bytes());
    buf.extend_from_slice(&tool_version.to_le_bytes());

    debug_assert_eq!(buf.len(), 32);
    buf
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nlist64_encode_decode_roundtrip() {
        let nlist = NList64 {
            strx: 42,
            typ: nlist_type::N_SECT | nlist_type::N_EXT,
            sect: 1,
            desc: 0,
            value: 0x1000,
        };

        let encoded = nlist.encode();
        assert_eq!(encoded.len(), NLIST_64_SIZE);

        let decoded = NList64::decode(&encoded);
        assert_eq!(nlist, decoded);
    }

    #[test]
    fn test_nlist64_type_queries() {
        let defined_ext = NList64 {
            strx: 0,
            typ: nlist_type::N_SECT | nlist_type::N_EXT,
            sect: 1,
            desc: 0,
            value: 0,
        };
        assert!(defined_ext.is_external());
        assert!(defined_ext.is_defined());
        assert!(!defined_ext.is_undefined());
        assert!(!defined_ext.is_private_extern());

        let undef = NList64 {
            strx: 0,
            typ: nlist_type::N_UNDF | nlist_type::N_EXT,
            sect: 0,
            desc: 0,
            value: 0,
        };
        assert!(undef.is_external());
        assert!(undef.is_undefined());
        assert!(!undef.is_defined());

        let private_ext = NList64 {
            strx: 0,
            typ: nlist_type::N_SECT | nlist_type::N_EXT | nlist_type::N_PEXT,
            sect: 1,
            desc: 0,
            value: 0,
        };
        assert!(private_ext.is_private_extern());
        assert!(private_ext.is_external());
    }

    #[test]
    fn test_symbol_table_basic() {
        let mut symtab = SymbolTable::new();

        // Add local symbol
        let idx0 = symtab.add_symbol("_local_func", 1, 0x0, false, false);
        assert_eq!(idx0, 0);

        // Add global symbol
        let idx1 = symtab.add_symbol("_main", 1, 0x100, true, false);
        assert_eq!(idx1, 1); // local(1) + external(0) before this one

        // Add undefined symbol
        let idx2 = symtab.add_undefined_symbol("_printf");
        assert_eq!(idx2, 2);

        assert_eq!(symtab.num_symbols(), 3);
        assert_eq!(symtab.num_local_symbols(), 1);
        assert_eq!(symtab.num_external_symbols(), 1);
        assert_eq!(symtab.num_undefined_symbols(), 1);
    }

    #[test]
    fn test_symbol_table_partitioning() {
        let mut symtab = SymbolTable::new();

        symtab.add_symbol("_local1", 1, 0, false, false);
        symtab.add_symbol("_local2", 1, 4, false, false);
        symtab.add_symbol("_global1", 1, 8, true, false);
        symtab.add_undefined_symbol("_extern1");

        let params = symtab.dysymtab_params();
        assert_eq!(params.ilocalsym, 0);
        assert_eq!(params.nlocalsym, 2);
        assert_eq!(params.iextdefsym, 2);
        assert_eq!(params.nextdefsym, 1);
        assert_eq!(params.iundefsym, 3);
        assert_eq!(params.nundefsym, 1);
    }

    #[test]
    fn test_symbol_table_write() {
        let mut symtab = SymbolTable::new();
        symtab.add_symbol("_main", 1, 0x1000, true, false);

        let bytes = symtab.write_symtab();
        assert_eq!(bytes.len(), NLIST_64_SIZE); // one symbol

        // Decode and verify
        let nlist = NList64::decode(bytes[..16].try_into().unwrap());
        assert_eq!(nlist.typ, nlist_type::N_SECT | nlist_type::N_EXT);
        assert_eq!(nlist.sect, 1);
        assert_eq!(nlist.value, 0x1000);
    }

    #[test]
    fn test_string_table() {
        let mut symtab = SymbolTable::new();
        symtab.add_symbol("_main", 1, 0, true, false);
        symtab.add_symbol("_foo", 1, 4, true, false);

        let strtab = symtab.write_strtab();
        // Should be: \0_main\0_foo\0
        assert_eq!(strtab[0], 0); // leading NUL
        assert_eq!(&strtab[1..6], b"_main");
        assert_eq!(strtab[6], 0); // NUL terminator
        assert_eq!(&strtab[7..11], b"_foo");
        assert_eq!(strtab[11], 0); // NUL terminator
    }

    #[test]
    fn test_string_table_dedup() {
        let mut symtab = SymbolTable::new();
        let idx1 = symtab.add_symbol("_main", 1, 0, true, false);
        let idx2 = symtab.add_symbol("_main", 2, 4, true, false);

        // Both symbols should reference the same string table entry
        let sym1 = symtab.get_symbol(idx1).unwrap();
        let sym2 = symtab.get_symbol(idx2).unwrap();
        assert_eq!(sym1.strx, sym2.strx);
    }

    #[test]
    fn test_get_symbol() {
        let mut symtab = SymbolTable::new();
        symtab.add_symbol("_local", 1, 0, false, false);
        symtab.add_symbol("_global", 1, 4, true, false);
        symtab.add_undefined_symbol("_undef");

        assert_eq!(
            symtab.symbol_category(0),
            Some(SymbolCategory::Local)
        );
        assert_eq!(
            symtab.symbol_category(1),
            Some(SymbolCategory::External)
        );
        assert_eq!(
            symtab.symbol_category(2),
            Some(SymbolCategory::Undefined)
        );
        assert_eq!(symtab.symbol_category(3), None);
    }

    #[test]
    fn test_dysymtab_encode() {
        let mut symtab = SymbolTable::new();
        symtab.add_symbol("_local", 1, 0, false, false);
        symtab.add_symbol("_global", 1, 4, true, false);
        symtab.add_undefined_symbol("_extern");

        let params = symtab.dysymtab_params();
        let cmd = params.encode_load_command();
        assert_eq!(cmd.len(), 80);

        // Verify LC_DYSYMTAB command type
        let cmd_type = u32::from_le_bytes([cmd[0], cmd[1], cmd[2], cmd[3]]);
        assert_eq!(cmd_type, 0x0B); // LC_DYSYMTAB

        // Verify cmdsize
        let cmdsize = u32::from_le_bytes([cmd[4], cmd[5], cmd[6], cmd[7]]);
        assert_eq!(cmdsize, 80);

        // Verify partition fields
        let ilocal = u32::from_le_bytes([cmd[8], cmd[9], cmd[10], cmd[11]]);
        let nlocal = u32::from_le_bytes([cmd[12], cmd[13], cmd[14], cmd[15]]);
        let iext = u32::from_le_bytes([cmd[16], cmd[17], cmd[18], cmd[19]]);
        let next = u32::from_le_bytes([cmd[20], cmd[21], cmd[22], cmd[23]]);
        let iundef = u32::from_le_bytes([cmd[24], cmd[25], cmd[26], cmd[27]]);
        let nundef = u32::from_le_bytes([cmd[28], cmd[29], cmd[30], cmd[31]]);

        assert_eq!(ilocal, 0);
        assert_eq!(nlocal, 1);
        assert_eq!(iext, 1);
        assert_eq!(next, 1);
        assert_eq!(iundef, 2);
        assert_eq!(nundef, 1);
    }

    #[test]
    fn test_build_version_macos() {
        // macOS 14.0.0
        let minos = 14 << 16;
        let sdk = 14 << 16;
        let cmd = encode_build_version(minos, sdk);

        assert_eq!(cmd.len(), 24);

        // LC_BUILD_VERSION = 0x32
        let cmd_type = u32::from_le_bytes([cmd[0], cmd[1], cmd[2], cmd[3]]);
        assert_eq!(cmd_type, 0x32);

        // cmdsize = 24
        let cmdsize = u32::from_le_bytes([cmd[4], cmd[5], cmd[6], cmd[7]]);
        assert_eq!(cmdsize, 24);

        // platform = PLATFORM_MACOS = 1
        let platform = u32::from_le_bytes([cmd[8], cmd[9], cmd[10], cmd[11]]);
        assert_eq!(platform, 1);

        // minos
        let minos_out = u32::from_le_bytes([cmd[12], cmd[13], cmd[14], cmd[15]]);
        assert_eq!(minos_out, 0x000E_0000);

        // ntools = 0
        let ntools = u32::from_le_bytes([cmd[20], cmd[21], cmd[22], cmd[23]]);
        assert_eq!(ntools, 0);
    }

    #[test]
    fn test_build_version_with_tool() {
        let minos = 14 << 16;
        let sdk = 14 << 16;
        let tool_ld = 3; // ld
        let tool_ver = 1 << 16; // 1.0.0
        let cmd = encode_build_version_with_tool(minos, sdk, tool_ld, tool_ver);

        assert_eq!(cmd.len(), 32);

        let ntools = u32::from_le_bytes([cmd[20], cmd[21], cmd[22], cmd[23]]);
        assert_eq!(ntools, 1);

        let tool = u32::from_le_bytes([cmd[24], cmd[25], cmd[26], cmd[27]]);
        assert_eq!(tool, 3); // ld

        let ver = u32::from_le_bytes([cmd[28], cmd[29], cmd[30], cmd[31]]);
        assert_eq!(ver, tool_ver);
    }
}
