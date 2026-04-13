// llvm2-codegen - DWARF debug information emission
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Reference: DWARF 4 spec (http://dwarfstd.org/doc/DWARF4.pdf)
// Reference: ~/llvm-project-ref/llvm/lib/CodeGen/AsmPrinter/DwarfDebug.cpp
// Reference: ~/llvm-project-ref/llvm/lib/CodeGen/AsmPrinter/DwarfUnit.cpp

//! DWARF debug information emission for LLVM2.
//!
//! Generates the standard DWARF debug sections used by debuggers (lldb, gdb)
//! to map machine code back to source constructs:
//!
//! - `__DWARF,__debug_abbrev` — abbreviation table defining DIE tag/attribute schemas
//! - `__DWARF,__debug_info` — the actual debug information entries (DIEs)
//! - `__DWARF,__debug_str` — string table for names referenced by `.debug_info`
//! - `__DWARF,__debug_line` — line number program (minimal: function start/end markers)
//!
//! # DWARF version
//!
//! We emit DWARF 4 for macOS compatibility. DWARF 4 is the last version
//! natively supported by all macOS toolchain components (dsymutil, lldb).
//!
//! # Section layout
//!
//! ```text
//! __debug_abbrev:
//!   Abbreviation 1: DW_TAG_compile_unit (has children)
//!   Abbreviation 2: DW_TAG_subprogram (no children or has children)
//!   Abbreviation 3: DW_TAG_formal_parameter (no children)
//!   Abbreviation 4: DW_TAG_base_type (no children)
//!   0x00 (terminator)
//!
//! __debug_info:
//!   Compilation Unit Header (11 bytes for 32-bit DWARF)
//!   DW_TAG_compile_unit DIE
//!     DW_TAG_subprogram DIEs (one per function)
//!       DW_TAG_formal_parameter DIEs (one per param)
//!     DW_TAG_base_type DIEs
//!   0x00 (end of children)
//!
//! __debug_str:
//!   "\0" (empty string at offset 0)
//!   "LLVM2 verified compiler backend\0"
//!   "<function names>\0"
//!   ...
//!
//! __debug_line:
//!   Line Number Program Header
//!   (minimal — function start/end markers only)
//! ```

use crate::macho::writer::MachOWriter;

// ---------------------------------------------------------------------------
// DWARF constants — Tags
// ---------------------------------------------------------------------------

/// DW_TAG_compile_unit — top-level compilation unit.
const DW_TAG_COMPILE_UNIT: u16 = 0x11;

/// DW_TAG_subprogram — function definition.
const DW_TAG_SUBPROGRAM: u16 = 0x2e;

/// DW_TAG_formal_parameter — function parameter.
const DW_TAG_FORMAL_PARAMETER: u16 = 0x05;

/// DW_TAG_base_type — primitive type (i32, f64, etc.).
const DW_TAG_BASE_TYPE: u16 = 0x24;

// ---------------------------------------------------------------------------
// DWARF constants — Attributes
// ---------------------------------------------------------------------------

/// DW_AT_name — name of the entity.
const DW_AT_NAME: u16 = 0x03;

/// DW_AT_low_pc — start address of the entity.
const DW_AT_LOW_PC: u16 = 0x11;

/// DW_AT_high_pc — end address or length of the entity.
const DW_AT_HIGH_PC: u16 = 0x12;

/// DW_AT_producer — compiler identification string.
const DW_AT_PRODUCER: u16 = 0x25;

/// DW_AT_language — source language.
const DW_AT_LANGUAGE: u16 = 0x13;

/// DW_AT_comp_dir — compilation directory.
const DW_AT_COMP_DIR: u16 = 0x1b;

/// DW_AT_stmt_list — offset into .debug_line section.
const DW_AT_STMT_LIST: u16 = 0x10;

/// DW_AT_byte_size — size of the type in bytes.
const DW_AT_BYTE_SIZE: u16 = 0x0b;

/// DW_AT_encoding — data encoding (signed, unsigned, float, etc.).
const DW_AT_ENCODING: u16 = 0x3e;

/// DW_AT_type — reference to a type DIE.
const DW_AT_TYPE: u16 = 0x49;

// ---------------------------------------------------------------------------
// DWARF constants — Forms
// ---------------------------------------------------------------------------

/// DW_FORM_addr — target address.
const DW_FORM_ADDR: u8 = 0x01;

/// DW_FORM_data1 — 1-byte unsigned integer.
const DW_FORM_DATA1: u8 = 0x0b;

/// DW_FORM_data2 — 2-byte unsigned integer.
const DW_FORM_DATA2: u8 = 0x05;

/// DW_FORM_data4 — 4-byte unsigned integer.
const DW_FORM_DATA4: u8 = 0x06;

/// DW_FORM_strp — offset into .debug_str section.
const DW_FORM_STRP: u8 = 0x0e;

/// DW_FORM_ref4 — 4-byte offset relative to compilation unit start.
const DW_FORM_REF4: u8 = 0x13;

/// DW_FORM_sec_offset — offset into another DWARF section.
const DW_FORM_SEC_OFFSET: u8 = 0x17;

// ---------------------------------------------------------------------------
// DWARF constants — Children flag
// ---------------------------------------------------------------------------

/// DW_CHILDREN_no — abbreviation has no children.
const DW_CHILDREN_NO: u8 = 0x00;

/// DW_CHILDREN_yes — abbreviation has children.
const DW_CHILDREN_YES: u8 = 0x01;

// ---------------------------------------------------------------------------
// DWARF constants — Language
// ---------------------------------------------------------------------------

/// DW_LANG_Rust.
const DW_LANG_RUST: u16 = 0x001c;

/// DW_LANG_C99.
const DW_LANG_C99: u16 = 0x000c;

// ---------------------------------------------------------------------------
// DWARF constants — Base type encoding (DW_ATE_*)
// ---------------------------------------------------------------------------

/// DW_ATE_signed — signed integer.
const DW_ATE_SIGNED: u8 = 0x05;

/// DW_ATE_unsigned — unsigned integer.
const DW_ATE_UNSIGNED: u8 = 0x07;

/// DW_ATE_float — floating point.
const DW_ATE_FLOAT: u8 = 0x04;

/// DW_ATE_boolean — boolean.
const DW_ATE_BOOLEAN: u8 = 0x02;

// ---------------------------------------------------------------------------
// DWARF version
// ---------------------------------------------------------------------------

/// DWARF format version (DWARF 4 for macOS compatibility).
const DWARF_VERSION: u16 = 4;

/// Address size on AArch64 (8 bytes = 64-bit).
const ADDRESS_SIZE: u8 = 8;

/// Section attribute: debug information section.
/// Reference: mach-o/loader.h S_ATTR_DEBUG
const S_ATTR_DEBUG: u32 = 0x0200_0000;

/// Producer string identifying this compiler.
const PRODUCER: &str = "LLVM2 verified compiler backend";

// ---------------------------------------------------------------------------
// Abbreviation IDs
// ---------------------------------------------------------------------------

/// Abbreviation ID for DW_TAG_compile_unit.
const ABBREV_COMPILE_UNIT: u32 = 1;

/// Abbreviation ID for DW_TAG_subprogram (with children for parameters).
const ABBREV_SUBPROGRAM: u32 = 2;

/// Abbreviation ID for DW_TAG_formal_parameter.
const ABBREV_FORMAL_PARAMETER: u32 = 3;

/// Abbreviation ID for DW_TAG_base_type.
const ABBREV_BASE_TYPE: u32 = 4;

/// Abbreviation ID for DW_TAG_subprogram (leaf — no children).
const ABBREV_SUBPROGRAM_LEAF: u32 = 5;

// ---------------------------------------------------------------------------
// Source language enum
// ---------------------------------------------------------------------------

/// Source language for the compilation unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceLanguage {
    /// Rust
    Rust,
    /// C99
    C99,
}

impl SourceLanguage {
    fn dwarf_encoding(self) -> u16 {
        match self {
            SourceLanguage::Rust => DW_LANG_RUST,
            SourceLanguage::C99 => DW_LANG_C99,
        }
    }
}

// ---------------------------------------------------------------------------
// BaseType — primitive type descriptors
// ---------------------------------------------------------------------------

/// A DWARF base type (DW_TAG_base_type) for primitive types.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BaseType {
    /// Type name (e.g., "i32", "f64").
    pub name: String,
    /// Size in bytes.
    pub byte_size: u8,
    /// DW_ATE_* encoding.
    pub encoding: u8,
}

impl BaseType {
    /// Create a signed integer type.
    pub fn signed(name: &str, byte_size: u8) -> Self {
        Self {
            name: name.to_string(),
            byte_size,
            encoding: DW_ATE_SIGNED,
        }
    }

    /// Create an unsigned integer type.
    pub fn unsigned(name: &str, byte_size: u8) -> Self {
        Self {
            name: name.to_string(),
            byte_size,
            encoding: DW_ATE_UNSIGNED,
        }
    }

    /// Create a floating-point type.
    pub fn float(name: &str, byte_size: u8) -> Self {
        Self {
            name: name.to_string(),
            byte_size,
            encoding: DW_ATE_FLOAT,
        }
    }

    /// Create a boolean type.
    pub fn boolean() -> Self {
        Self {
            name: "bool".to_string(),
            byte_size: 1,
            encoding: DW_ATE_BOOLEAN,
        }
    }

    /// Standard i8 type.
    pub fn i8() -> Self { Self::signed("i8", 1) }
    /// Standard i16 type.
    pub fn i16() -> Self { Self::signed("i16", 2) }
    /// Standard i32 type.
    pub fn i32() -> Self { Self::signed("i32", 4) }
    /// Standard i64 type.
    pub fn i64() -> Self { Self::signed("i64", 8) }
    /// Standard f32 type.
    pub fn f32() -> Self { Self::float("f32", 4) }
    /// Standard f64 type.
    pub fn f64() -> Self { Self::float("f64", 8) }
}

// ---------------------------------------------------------------------------
// FunctionParam — parameter descriptor
// ---------------------------------------------------------------------------

/// A function parameter for debug info.
#[derive(Debug, Clone)]
pub struct FunctionParam {
    /// Parameter name.
    pub name: String,
    /// Index into the base types array (set during emission).
    pub type_index: usize,
}

// ---------------------------------------------------------------------------
// FunctionDebugInfo — per-function debug metadata
// ---------------------------------------------------------------------------

/// Debug information for a single function.
#[derive(Debug, Clone)]
pub struct FunctionDebugInfo {
    /// Function name (without Mach-O '_' prefix).
    pub name: String,
    /// Start address of the function (offset within __text section).
    pub low_pc: u64,
    /// Size of the function in bytes.
    pub size: u32,
    /// Function parameters.
    pub params: Vec<FunctionParam>,
}

// ---------------------------------------------------------------------------
// DwarfDebugInfo — collects all debug information for emission
// ---------------------------------------------------------------------------

/// Top-level DWARF debug information builder.
///
/// Collects compilation unit metadata, function info, and type info,
/// then emits the four DWARF sections.
///
/// # Usage
///
/// ```no_run
/// use llvm2_codegen::dwarf_info::{DwarfDebugInfo, SourceLanguage, FunctionDebugInfo, BaseType};
///
/// let mut dbg = DwarfDebugInfo::new(
///     "test.rs",
///     "/home/user/project",
///     SourceLanguage::Rust,
/// );
/// dbg.add_function(FunctionDebugInfo {
///     name: "add".to_string(),
///     low_pc: 0,
///     size: 8,
///     params: vec![],
/// });
/// dbg.add_base_type(BaseType::i32());
///
/// let abbrev = dbg.emit_abbrev();
/// let str_tab = dbg.emit_debug_str();
/// let info = dbg.emit_debug_info();
/// let line = dbg.emit_debug_line();
/// ```
#[derive(Debug, Clone)]
pub struct DwarfDebugInfo {
    /// Source file name.
    pub file_name: String,
    /// Compilation directory.
    pub comp_dir: String,
    /// Source language.
    pub language: SourceLanguage,
    /// Functions in this compilation unit.
    pub functions: Vec<FunctionDebugInfo>,
    /// Base types used by this compilation unit.
    pub base_types: Vec<BaseType>,
}

impl DwarfDebugInfo {
    /// Create a new DWARF debug info builder.
    pub fn new(file_name: &str, comp_dir: &str, language: SourceLanguage) -> Self {
        Self {
            file_name: file_name.to_string(),
            comp_dir: comp_dir.to_string(),
            language,
            functions: Vec::new(),
            base_types: Vec::new(),
        }
    }

    /// Add a function to the debug info.
    pub fn add_function(&mut self, func: FunctionDebugInfo) {
        self.functions.push(func);
    }

    /// Add a base type to the debug info.
    pub fn add_base_type(&mut self, ty: BaseType) {
        self.base_types.push(ty);
    }

    /// Add standard tMIR base types (i8, i16, i32, i64, f32, f64, bool).
    pub fn add_standard_types(&mut self) {
        self.base_types.push(BaseType::i8());
        self.base_types.push(BaseType::i16());
        self.base_types.push(BaseType::i32());
        self.base_types.push(BaseType::i64());
        self.base_types.push(BaseType::f32());
        self.base_types.push(BaseType::f64());
        self.base_types.push(BaseType::boolean());
    }

    // -----------------------------------------------------------------------
    // .debug_str — String table
    // -----------------------------------------------------------------------

    /// Build the string table, returning (bytes, offset_map).
    ///
    /// The offset map maps each string to its byte offset in the table.
    /// String table starts with a null byte (offset 0 = empty string).
    fn build_string_table(&self) -> (Vec<u8>, StringTable) {
        let mut table = StringTable::new();

        // Register all strings we'll need
        table.insert(PRODUCER);
        table.insert(&self.file_name);
        table.insert(&self.comp_dir);

        for func in &self.functions {
            table.insert(&func.name);
            for param in &func.params {
                table.insert(&param.name);
            }
        }

        for ty in &self.base_types {
            table.insert(&ty.name);
        }

        (table.to_bytes(), table)
    }

    /// Emit the `.debug_str` section bytes.
    pub fn emit_debug_str(&self) -> Vec<u8> {
        let (bytes, _) = self.build_string_table();
        bytes
    }

    // -----------------------------------------------------------------------
    // .debug_abbrev — Abbreviation table
    // -----------------------------------------------------------------------

    /// Emit the `.debug_abbrev` section bytes.
    ///
    /// Defines the abbreviation entries used by `.debug_info`:
    ///
    /// 1. `DW_TAG_compile_unit` — has children
    ///    - DW_AT_producer (DW_FORM_strp)
    ///    - DW_AT_language (DW_FORM_data2)
    ///    - DW_AT_name (DW_FORM_strp)
    ///    - DW_AT_comp_dir (DW_FORM_strp)
    ///    - DW_AT_low_pc (DW_FORM_addr)
    ///    - DW_AT_high_pc (DW_FORM_data4) — DWARF4: length form
    ///    - DW_AT_stmt_list (DW_FORM_sec_offset)
    ///
    /// 2. `DW_TAG_subprogram` (with children)
    ///    - DW_AT_name (DW_FORM_strp)
    ///    - DW_AT_low_pc (DW_FORM_addr)
    ///    - DW_AT_high_pc (DW_FORM_data4)
    ///
    /// 3. `DW_TAG_formal_parameter`
    ///    - DW_AT_name (DW_FORM_strp)
    ///    - DW_AT_type (DW_FORM_ref4)
    ///
    /// 4. `DW_TAG_base_type`
    ///    - DW_AT_name (DW_FORM_strp)
    ///    - DW_AT_byte_size (DW_FORM_data1)
    ///    - DW_AT_encoding (DW_FORM_data1)
    ///
    /// 5. `DW_TAG_subprogram` (leaf — no children)
    ///    - DW_AT_name (DW_FORM_strp)
    ///    - DW_AT_low_pc (DW_FORM_addr)
    ///    - DW_AT_high_pc (DW_FORM_data4)
    pub fn emit_abbrev(&self) -> Vec<u8> {
        let mut data = Vec::new();

        // Abbreviation 1: DW_TAG_compile_unit (has children)
        encode_uleb128(ABBREV_COMPILE_UNIT as u64, &mut data);
        encode_uleb128(DW_TAG_COMPILE_UNIT as u64, &mut data);
        data.push(DW_CHILDREN_YES);
        // Attributes
        emit_abbrev_attr(&mut data, DW_AT_PRODUCER, DW_FORM_STRP);
        emit_abbrev_attr(&mut data, DW_AT_LANGUAGE, DW_FORM_DATA2);
        emit_abbrev_attr(&mut data, DW_AT_NAME, DW_FORM_STRP);
        emit_abbrev_attr(&mut data, DW_AT_COMP_DIR, DW_FORM_STRP);
        emit_abbrev_attr(&mut data, DW_AT_LOW_PC, DW_FORM_ADDR);
        emit_abbrev_attr(&mut data, DW_AT_HIGH_PC, DW_FORM_DATA4);
        emit_abbrev_attr(&mut data, DW_AT_STMT_LIST, DW_FORM_SEC_OFFSET);
        // End of attributes
        data.push(0);
        data.push(0);

        // Abbreviation 2: DW_TAG_subprogram (has children — for functions with params)
        encode_uleb128(ABBREV_SUBPROGRAM as u64, &mut data);
        encode_uleb128(DW_TAG_SUBPROGRAM as u64, &mut data);
        data.push(DW_CHILDREN_YES);
        emit_abbrev_attr(&mut data, DW_AT_NAME, DW_FORM_STRP);
        emit_abbrev_attr(&mut data, DW_AT_LOW_PC, DW_FORM_ADDR);
        emit_abbrev_attr(&mut data, DW_AT_HIGH_PC, DW_FORM_DATA4);
        data.push(0);
        data.push(0);

        // Abbreviation 3: DW_TAG_formal_parameter
        encode_uleb128(ABBREV_FORMAL_PARAMETER as u64, &mut data);
        encode_uleb128(DW_TAG_FORMAL_PARAMETER as u64, &mut data);
        data.push(DW_CHILDREN_NO);
        emit_abbrev_attr(&mut data, DW_AT_NAME, DW_FORM_STRP);
        emit_abbrev_attr(&mut data, DW_AT_TYPE, DW_FORM_REF4);
        data.push(0);
        data.push(0);

        // Abbreviation 4: DW_TAG_base_type
        encode_uleb128(ABBREV_BASE_TYPE as u64, &mut data);
        encode_uleb128(DW_TAG_BASE_TYPE as u64, &mut data);
        data.push(DW_CHILDREN_NO);
        emit_abbrev_attr(&mut data, DW_AT_NAME, DW_FORM_STRP);
        emit_abbrev_attr(&mut data, DW_AT_BYTE_SIZE, DW_FORM_DATA1);
        emit_abbrev_attr(&mut data, DW_AT_ENCODING, DW_FORM_DATA1);
        data.push(0);
        data.push(0);

        // Abbreviation 5: DW_TAG_subprogram (leaf — no children)
        encode_uleb128(ABBREV_SUBPROGRAM_LEAF as u64, &mut data);
        encode_uleb128(DW_TAG_SUBPROGRAM as u64, &mut data);
        data.push(DW_CHILDREN_NO);
        emit_abbrev_attr(&mut data, DW_AT_NAME, DW_FORM_STRP);
        emit_abbrev_attr(&mut data, DW_AT_LOW_PC, DW_FORM_ADDR);
        emit_abbrev_attr(&mut data, DW_AT_HIGH_PC, DW_FORM_DATA4);
        data.push(0);
        data.push(0);

        // Terminator: null abbreviation
        data.push(0);

        data
    }

    // -----------------------------------------------------------------------
    // .debug_info — Debug information entries
    // -----------------------------------------------------------------------

    /// Emit the `.debug_info` section bytes.
    ///
    /// Layout:
    /// - Compilation Unit Header (11 bytes for 32-bit DWARF 4)
    ///   - u32: unit_length (length of rest of CU, excluding this field)
    ///   - u16: version (4)
    ///   - u32: debug_abbrev_offset (0)
    ///   - u8: address_size (8)
    /// - DW_TAG_compile_unit DIE
    /// - DW_TAG_subprogram DIEs
    /// - DW_TAG_base_type DIEs
    /// - 0x00 (end of compile_unit children)
    pub fn emit_debug_info(&self) -> Vec<u8> {
        let (_, str_table) = self.build_string_table();

        // Build the body first (everything after the unit_length field),
        // then prepend the length.
        let mut body = Vec::new();

        // CU header (after unit_length)
        body.extend_from_slice(&DWARF_VERSION.to_le_bytes()); // version
        body.extend_from_slice(&0u32.to_le_bytes());          // debug_abbrev_offset
        body.push(ADDRESS_SIZE);                               // address_size

        // Compute the compilation unit's address range from functions.
        let (cu_low_pc, cu_high_pc_length) = self.compute_cu_range();

        // DW_TAG_compile_unit DIE (abbreviation 1)
        encode_uleb128(ABBREV_COMPILE_UNIT as u64, &mut body);
        // DW_AT_producer (strp)
        body.extend_from_slice(&str_table.offset_of(PRODUCER).to_le_bytes());
        // DW_AT_language (data2)
        body.extend_from_slice(&self.language.dwarf_encoding().to_le_bytes());
        // DW_AT_name (strp)
        body.extend_from_slice(&str_table.offset_of(&self.file_name).to_le_bytes());
        // DW_AT_comp_dir (strp)
        body.extend_from_slice(&str_table.offset_of(&self.comp_dir).to_le_bytes());
        // DW_AT_low_pc (addr)
        body.extend_from_slice(&cu_low_pc.to_le_bytes());
        // DW_AT_high_pc (data4) — DWARF4 length form
        body.extend_from_slice(&cu_high_pc_length.to_le_bytes());
        // DW_AT_stmt_list (sec_offset) — offset into .debug_line (always 0)
        body.extend_from_slice(&0u32.to_le_bytes());

        // We need to track where base type DIEs start so subprogram params
        // can reference them via DW_FORM_ref4 (CU-relative offset).
        // Strategy: emit subprograms first, then base types. Record base type
        // offsets as we emit them.
        //
        // The CU-relative offset includes the CU header (7 bytes for DWARF 4:
        // version(2) + abbrev_offset(4) + addr_size(1)). But DW_FORM_ref4
        // is relative to the start of the CU *header* (after unit_length),
        // which is the start of `body`.

        // Emit subprogram DIEs
        for func in &self.functions {
            let has_params = !func.params.is_empty();
            let abbrev = if has_params {
                ABBREV_SUBPROGRAM
            } else {
                ABBREV_SUBPROGRAM_LEAF
            };

            encode_uleb128(abbrev as u64, &mut body);
            // DW_AT_name (strp)
            body.extend_from_slice(&str_table.offset_of(&func.name).to_le_bytes());
            // DW_AT_low_pc (addr)
            body.extend_from_slice(&func.low_pc.to_le_bytes());
            // DW_AT_high_pc (data4) — length
            body.extend_from_slice(&func.size.to_le_bytes());

            if has_params {
                // Emit formal parameter DIEs
                for param in &func.params {
                    encode_uleb128(ABBREV_FORMAL_PARAMETER as u64, &mut body);
                    // DW_AT_name (strp)
                    body.extend_from_slice(&str_table.offset_of(&param.name).to_le_bytes());
                    // DW_AT_type (ref4) — CU-relative offset to base type DIE
                    // We use a placeholder here; the caller should set type_index
                    // to the correct base type. We'll fix up below.
                    // For now, emit 0 as placeholder.
                    body.extend_from_slice(&0u32.to_le_bytes());
                }
                // End of subprogram children
                body.push(0);
            }
        }

        // Record offset where base types start (for type references)
        let base_type_offsets: Vec<u32> = {
            let mut offsets = Vec::new();
            let mut offset = body.len() as u32;
            for _ty in &self.base_types {
                offsets.push(offset);
                // Size of a base type DIE: uleb128(abbrev) + strp(4) + data1(1) + data1(1)
                let mut tmp = Vec::new();
                encode_uleb128(ABBREV_BASE_TYPE as u64, &mut tmp);
                offset += tmp.len() as u32 + 4 + 1 + 1;
            }
            offsets
        };

        // Emit base type DIEs
        for ty in &self.base_types {
            encode_uleb128(ABBREV_BASE_TYPE as u64, &mut body);
            // DW_AT_name (strp)
            body.extend_from_slice(&str_table.offset_of(&ty.name).to_le_bytes());
            // DW_AT_byte_size (data1)
            body.push(ty.byte_size);
            // DW_AT_encoding (data1)
            body.push(ty.encoding);
        }

        // End of compile_unit children
        body.push(0);

        // Now fix up the DW_AT_type references in formal_parameter DIEs.
        // We need to go back and patch the ref4 fields.
        if !self.base_types.is_empty() {
            self.fixup_type_refs(&mut body, &str_table, &base_type_offsets);
        }

        // Prepend unit_length (4 bytes, length of everything after this field)
        let unit_length = body.len() as u32;
        let mut result = Vec::with_capacity(4 + body.len());
        result.extend_from_slice(&unit_length.to_le_bytes());
        result.extend(body);

        result
    }

    /// Fix up DW_AT_type references in formal parameter DIEs.
    ///
    /// Walks the body bytes looking for formal parameter DIEs and patches
    /// their type reference (DW_FORM_ref4) to point to the correct base type.
    fn fixup_type_refs(
        &self,
        body: &mut [u8],
        _str_table: &StringTable,
        base_type_offsets: &[u32],
    ) {
        // Skip CU header: version(2) + abbrev_offset(4) + addr_size(1) = 7
        // Skip compile_unit DIE: abbrev(1+) + producer(4) + language(2) + name(4)
        //   + comp_dir(4) + low_pc(8) + high_pc(4) + stmt_list(4)
        // Then for each subprogram with children, find formal_parameter DIEs.

        // Rather than parsing DWARF, we use a simpler approach: scan for each
        // parameter and compute its offset directly.
        let mut offset = 7; // skip CU header

        // Skip compile_unit DIE abbreviation code
        let (_, len) = decode_uleb128(&body[offset..]);
        offset += len;

        // Skip compile_unit attributes:
        // producer(4) + language(2) + name(4) + comp_dir(4) + low_pc(8) + high_pc(4) + stmt_list(4)
        offset += 4 + 2 + 4 + 4 + 8 + 4 + 4; // = 30

        // Now we're at the subprogram DIEs
        for func in &self.functions {
            let has_params = !func.params.is_empty();

            // Skip subprogram abbreviation code
            let (_, len) = decode_uleb128(&body[offset..]);
            offset += len;

            // Skip subprogram attributes: name(4) + low_pc(8) + high_pc(4)
            offset += 4 + 8 + 4; // = 16

            if has_params {
                for param in &func.params {
                    // Skip formal_parameter abbreviation code
                    let (_, len) = decode_uleb128(&body[offset..]);
                    offset += len;

                    // Skip name (strp, 4 bytes)
                    offset += 4;

                    // Patch type reference (ref4, 4 bytes) — currently 0
                    let type_idx = param.type_index;
                    if type_idx < base_type_offsets.len() {
                        let type_offset = base_type_offsets[type_idx];
                        body[offset..offset + 4]
                            .copy_from_slice(&type_offset.to_le_bytes());
                    }
                    offset += 4;
                }
                // Skip null terminator for subprogram children
                offset += 1;
            }
        }
    }

    /// Compute the compilation unit's address range from its functions.
    ///
    /// Returns (low_pc, high_pc_length) where high_pc_length is the total
    /// span in bytes (DWARF 4 DW_FORM_data4 form for DW_AT_high_pc).
    fn compute_cu_range(&self) -> (u64, u32) {
        if self.functions.is_empty() {
            return (0, 0);
        }

        let low = self
            .functions
            .iter()
            .map(|f| f.low_pc)
            .min()
            .unwrap_or(0);

        let high = self
            .functions
            .iter()
            .map(|f| f.low_pc + f.size as u64)
            .max()
            .unwrap_or(0);

        (low, (high - low) as u32)
    }

    // -----------------------------------------------------------------------
    // .debug_line — Line number program (minimal)
    // -----------------------------------------------------------------------

    /// Emit a minimal `.debug_line` section.
    ///
    /// For the initial implementation this emits a valid but minimal line
    /// table: just the header with file/directory tables and a trivial
    /// line program that sets addresses at function boundaries. Full
    /// source-line mapping is a follow-up task.
    ///
    /// DWARF 4 line program header layout:
    /// - u32: unit_length
    /// - u16: version (4)
    /// - u32: header_length
    /// - u8: minimum_instruction_length (4 for AArch64)
    /// - u8: maximum_operations_per_instruction (1)
    /// - u8: default_is_stmt (1)
    /// - i8: line_base (-5)
    /// - u8: line_range (14)
    /// - u8: opcode_base (13)
    /// - u8[12]: standard_opcode_lengths
    /// - include directories (null-terminated list)
    /// - file names (null-terminated list)
    /// - line program opcodes
    pub fn emit_debug_line(&self) -> Vec<u8> {
        // Build header body first (everything after unit_length)
        let mut header_body = Vec::new();

        // version
        header_body.extend_from_slice(&4u16.to_le_bytes());

        // We'll fill in header_length after building the header content
        let header_length_pos = header_body.len();
        header_body.extend_from_slice(&0u32.to_le_bytes()); // placeholder

        let header_content_start = header_body.len();

        // minimum_instruction_length (4 for AArch64)
        header_body.push(4);
        // maximum_operations_per_instruction (1 for non-VLIW)
        header_body.push(1);
        // default_is_stmt
        header_body.push(1);
        // line_base
        header_body.push((-5i8) as u8);
        // line_range
        header_body.push(14);
        // opcode_base (13 = standard opcodes 1-12)
        header_body.push(13);
        // standard_opcode_lengths for opcodes 1-12
        // (from DWARF 4 spec, Figure 37)
        header_body.extend_from_slice(&[0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1]);

        // Include directories (terminated by empty string = 0 byte)
        // Just the compilation directory
        header_body.extend_from_slice(self.comp_dir.as_bytes());
        header_body.push(0); // null terminate the directory string
        header_body.push(0); // end of directory list

        // File names (terminated by empty entry = 0 byte)
        // Each entry: name (null-terminated), directory index (ULEB128),
        //             modification time (ULEB128), file length (ULEB128)
        header_body.extend_from_slice(self.file_name.as_bytes());
        header_body.push(0); // null terminate file name
        encode_uleb128(1, &mut header_body); // directory index = 1 (first dir)
        encode_uleb128(0, &mut header_body); // modification time (unknown)
        encode_uleb128(0, &mut header_body); // file length (unknown)
        header_body.push(0); // end of file list

        // Patch header_length
        let header_content_len =
            (header_body.len() - header_content_start) as u32;
        header_body[header_length_pos..header_length_pos + 4]
            .copy_from_slice(&header_content_len.to_le_bytes());

        // Line number program opcodes
        // Minimal: for each function, set address and emit a row.
        //
        // DW_LNS_set_file = 4 (operand: ULEB128 file index)
        // DW_LNE_set_address = 0x00, len, 0x02, addr
        // DW_LNS_copy = 1
        // DW_LNE_end_sequence = 0x00, 0x01, 0x01

        // Set file to index 1
        header_body.push(4); // DW_LNS_set_file
        encode_uleb128(1, &mut header_body);

        for func in &self.functions {
            // Extended opcode: DW_LNE_set_address
            header_body.push(0); // extended opcode marker
            encode_uleb128(1 + ADDRESS_SIZE as u64, &mut header_body); // length
            header_body.push(0x02); // DW_LNE_set_address
            header_body.extend_from_slice(&func.low_pc.to_le_bytes());

            // DW_LNS_copy — emit a row
            header_body.push(1);

            // Advance to function end
            // Extended opcode: DW_LNE_set_address for end
            header_body.push(0);
            encode_uleb128(1 + ADDRESS_SIZE as u64, &mut header_body);
            header_body.push(0x02); // DW_LNE_set_address
            let end_pc = func.low_pc + func.size as u64;
            header_body.extend_from_slice(&end_pc.to_le_bytes());
        }

        // End sequence
        header_body.push(0);    // extended opcode marker
        header_body.push(0x01); // length = 1
        header_body.push(0x01); // DW_LNE_end_sequence

        // Prepend unit_length
        let unit_length = header_body.len() as u32;
        let mut result = Vec::with_capacity(4 + header_body.len());
        result.extend_from_slice(&unit_length.to_le_bytes());
        result.extend(header_body);

        result
    }
}

// ---------------------------------------------------------------------------
// StringTable — helper for .debug_str
// ---------------------------------------------------------------------------

/// A DWARF string table builder.
///
/// Deduplicates strings and tracks their byte offsets. The table always
/// starts with a null byte at offset 0 (the empty string).
#[derive(Debug, Clone)]
struct StringTable {
    /// The raw bytes of the string table.
    data: Vec<u8>,
    /// Map from string content to byte offset.
    offsets: std::collections::HashMap<String, u32>,
}

impl StringTable {
    fn new() -> Self {
        let mut table = Self {
            data: vec![0], // offset 0 = empty string (null byte)
            offsets: std::collections::HashMap::new(),
        };
        // Pre-register the empty string
        table.offsets.insert(String::new(), 0);
        table
    }

    /// Insert a string into the table. Returns its byte offset.
    fn insert(&mut self, s: &str) -> u32 {
        if let Some(&offset) = self.offsets.get(s) {
            return offset;
        }
        let offset = self.data.len() as u32;
        self.data.extend_from_slice(s.as_bytes());
        self.data.push(0); // null terminator
        self.offsets.insert(s.to_string(), offset);
        offset
    }

    /// Get the byte offset of a previously inserted string.
    ///
    /// Panics if the string was not inserted.
    fn offset_of(&self, s: &str) -> u32 {
        self.offsets[s]
    }

    /// Serialize the string table to bytes.
    fn to_bytes(&self) -> Vec<u8> {
        self.data.clone()
    }
}

// ---------------------------------------------------------------------------
// LEB128 encoding helpers
// ---------------------------------------------------------------------------

/// Encode a value as ULEB128 (unsigned LEB128).
fn encode_uleb128(mut value: u64, out: &mut Vec<u8>) {
    loop {
        let mut byte = (value & 0x7F) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
        }
        out.push(byte);
        if value == 0 {
            break;
        }
    }
}

/// Decode a ULEB128 value from a byte slice.
/// Returns (value, bytes_consumed).
fn decode_uleb128(data: &[u8]) -> (u64, usize) {
    let mut value: u64 = 0;
    let mut shift = 0;
    let mut i = 0;
    loop {
        let byte = data[i];
        value |= ((byte & 0x7F) as u64) << shift;
        i += 1;
        if byte & 0x80 == 0 {
            break;
        }
        shift += 7;
    }
    (value, i)
}

// ---------------------------------------------------------------------------
// Abbreviation attribute helper
// ---------------------------------------------------------------------------

/// Emit a single abbreviation attribute (name, form) pair.
fn emit_abbrev_attr(data: &mut Vec<u8>, name: u16, form: u8) {
    encode_uleb128(name as u64, data);
    encode_uleb128(form as u64, data);
}

// ---------------------------------------------------------------------------
// MachOWriter integration
// ---------------------------------------------------------------------------

/// Add DWARF debug sections to a MachOWriter.
///
/// Emits four sections in the `__DWARF` segment:
/// - `__debug_abbrev`
/// - `__debug_info`
/// - `__debug_str`
/// - `__debug_line`
///
/// All sections use `S_ATTR_DEBUG` and 1-byte alignment (no alignment
/// requirements for DWARF sections).
///
/// # Returns
///
/// Tuple of (abbrev_index, info_index, str_index, line_index) — the
/// 0-based section indices, or `None` if the debug info has no functions.
pub fn add_debug_info_to_writer(
    writer: &mut MachOWriter,
    debug_info: &DwarfDebugInfo,
) -> Option<(usize, usize, usize, usize)> {
    if debug_info.functions.is_empty() {
        return None;
    }

    let abbrev_data = debug_info.emit_abbrev();
    let info_data = debug_info.emit_debug_info();
    let str_data = debug_info.emit_debug_str();
    let line_data = debug_info.emit_debug_line();

    let abbrev_idx = writer.add_custom_section(
        b"__debug_abbrev",
        b"__DWARF",
        &abbrev_data,
        0, // 2^0 = 1-byte alignment
        S_ATTR_DEBUG,
    );

    let info_idx = writer.add_custom_section(
        b"__debug_info",
        b"__DWARF",
        &info_data,
        0,
        S_ATTR_DEBUG,
    );

    let str_idx = writer.add_custom_section(
        b"__debug_str",
        b"__DWARF",
        &str_data,
        0,
        S_ATTR_DEBUG,
    );

    let line_idx = writer.add_custom_section(
        b"__debug_line",
        b"__DWARF",
        &line_data,
        0,
        S_ATTR_DEBUG,
    );

    Some((abbrev_idx, info_idx, str_idx, line_idx))
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_simple_debug_info() -> DwarfDebugInfo {
        let mut dbg = DwarfDebugInfo::new("test.rs", "/home/user/project", SourceLanguage::Rust);
        dbg.add_function(FunctionDebugInfo {
            name: "add".to_string(),
            low_pc: 0,
            size: 8,
            params: vec![],
        });
        dbg.add_base_type(BaseType::i32());
        dbg
    }

    fn make_multi_function_debug_info() -> DwarfDebugInfo {
        let mut dbg = DwarfDebugInfo::new("lib.rs", "/home/user/project", SourceLanguage::Rust);
        dbg.add_function(FunctionDebugInfo {
            name: "add".to_string(),
            low_pc: 0,
            size: 8,
            params: vec![
                FunctionParam { name: "a".to_string(), type_index: 0 },
                FunctionParam { name: "b".to_string(), type_index: 0 },
            ],
        });
        dbg.add_function(FunctionDebugInfo {
            name: "sub".to_string(),
            low_pc: 8,
            size: 12,
            params: vec![],
        });
        dbg.add_base_type(BaseType::i32());
        dbg.add_base_type(BaseType::i64());
        dbg
    }

    // --- String table tests ---

    #[test]
    fn test_string_table_starts_with_null() {
        let table = StringTable::new();
        let bytes = table.to_bytes();
        assert_eq!(bytes[0], 0);
        assert_eq!(bytes.len(), 1);
    }

    #[test]
    fn test_string_table_insert_and_lookup() {
        let mut table = StringTable::new();
        let off1 = table.insert("hello");
        let off2 = table.insert("world");

        assert_eq!(off1, 1); // after the initial null byte
        assert_eq!(off2, 7); // "hello\0" = 6 bytes, so "world" starts at 7

        // Deduplication
        let off1_again = table.insert("hello");
        assert_eq!(off1_again, off1);

        assert_eq!(table.offset_of("hello"), 1);
        assert_eq!(table.offset_of("world"), 7);
    }

    #[test]
    fn test_string_table_bytes_contain_strings() {
        let mut table = StringTable::new();
        table.insert("test");
        let bytes = table.to_bytes();

        // bytes: [0, 't', 'e', 's', 't', 0]
        assert_eq!(bytes.len(), 6);
        assert_eq!(&bytes[1..5], b"test");
        assert_eq!(bytes[5], 0);
    }

    // --- Abbreviation table tests ---

    #[test]
    fn test_abbrev_not_empty() {
        let dbg = make_simple_debug_info();
        let abbrev = dbg.emit_abbrev();
        assert!(!abbrev.is_empty());
    }

    #[test]
    fn test_abbrev_starts_with_abbreviation_1() {
        let dbg = make_simple_debug_info();
        let abbrev = dbg.emit_abbrev();
        // First byte should be ULEB128 encoding of 1
        assert_eq!(abbrev[0], 1);
    }

    #[test]
    fn test_abbrev_ends_with_null_terminator() {
        let dbg = make_simple_debug_info();
        let abbrev = dbg.emit_abbrev();
        // Last byte should be 0 (null abbreviation)
        assert_eq!(*abbrev.last().unwrap(), 0);
    }

    #[test]
    fn test_abbrev_contains_compile_unit_tag() {
        let dbg = make_simple_debug_info();
        let abbrev = dbg.emit_abbrev();
        // Abbreviation 1: tag = DW_TAG_compile_unit = 0x11
        // After ULEB128(1) = [0x01], we expect ULEB128(0x11) = [0x11]
        assert_eq!(abbrev[1], 0x11);
    }

    #[test]
    fn test_abbrev_compile_unit_has_children() {
        let dbg = make_simple_debug_info();
        let abbrev = dbg.emit_abbrev();
        // After ULEB128(1) and ULEB128(0x11), the children flag
        assert_eq!(abbrev[2], DW_CHILDREN_YES);
    }

    // --- Debug info section tests ---

    #[test]
    fn test_debug_info_not_empty() {
        let dbg = make_simple_debug_info();
        let info = dbg.emit_debug_info();
        assert!(!info.is_empty());
    }

    #[test]
    fn test_debug_info_starts_with_unit_length() {
        let dbg = make_simple_debug_info();
        let info = dbg.emit_debug_info();

        let unit_length = u32::from_le_bytes(info[0..4].try_into().unwrap());
        assert_eq!(unit_length as usize, info.len() - 4);
    }

    #[test]
    fn test_debug_info_version_is_4() {
        let dbg = make_simple_debug_info();
        let info = dbg.emit_debug_info();

        // After unit_length (4 bytes), version is next (2 bytes)
        let version = u16::from_le_bytes(info[4..6].try_into().unwrap());
        assert_eq!(version, 4);
    }

    #[test]
    fn test_debug_info_address_size_is_8() {
        let dbg = make_simple_debug_info();
        let info = dbg.emit_debug_info();

        // After unit_length(4) + version(2) + abbrev_offset(4) = offset 10
        assert_eq!(info[10], 8);
    }

    #[test]
    fn test_debug_info_first_die_is_compile_unit() {
        let dbg = make_simple_debug_info();
        let info = dbg.emit_debug_info();

        // After CU header: unit_length(4) + version(2) + abbrev_offset(4) + addr_size(1) = 11
        // First DIE should use abbreviation 1 (compile_unit)
        let (abbrev_code, _) = decode_uleb128(&info[11..]);
        assert_eq!(abbrev_code, ABBREV_COMPILE_UNIT as u64);
    }

    #[test]
    fn test_debug_info_ends_with_null() {
        let dbg = make_simple_debug_info();
        let info = dbg.emit_debug_info();

        // Last byte should be 0 (end of compile_unit children)
        assert_eq!(*info.last().unwrap(), 0);
    }

    #[test]
    fn test_debug_info_with_multiple_functions() {
        let dbg = make_multi_function_debug_info();
        let info = dbg.emit_debug_info();

        // Should be valid (non-empty, correct length field)
        let unit_length = u32::from_le_bytes(info[0..4].try_into().unwrap());
        assert_eq!(unit_length as usize, info.len() - 4);
    }

    #[test]
    fn test_debug_info_cu_range() {
        let dbg = make_multi_function_debug_info();
        let (low, len) = dbg.compute_cu_range();
        // Functions: add at 0 size 8, sub at 8 size 12
        // Range: 0 .. 20, so low=0, length=20
        assert_eq!(low, 0);
        assert_eq!(len, 20);
    }

    #[test]
    fn test_debug_info_empty_functions() {
        let dbg = DwarfDebugInfo::new("empty.rs", "/tmp", SourceLanguage::Rust);
        let (low, len) = dbg.compute_cu_range();
        assert_eq!(low, 0);
        assert_eq!(len, 0);
    }

    // --- Debug line section tests ---

    #[test]
    fn test_debug_line_not_empty() {
        let dbg = make_simple_debug_info();
        let line = dbg.emit_debug_line();
        assert!(!line.is_empty());
    }

    #[test]
    fn test_debug_line_starts_with_unit_length() {
        let dbg = make_simple_debug_info();
        let line = dbg.emit_debug_line();

        let unit_length = u32::from_le_bytes(line[0..4].try_into().unwrap());
        assert_eq!(unit_length as usize, line.len() - 4);
    }

    #[test]
    fn test_debug_line_version_is_4() {
        let dbg = make_simple_debug_info();
        let line = dbg.emit_debug_line();

        // After unit_length (4 bytes)
        let version = u16::from_le_bytes(line[4..6].try_into().unwrap());
        assert_eq!(version, 4);
    }

    #[test]
    fn test_debug_line_ends_with_end_sequence() {
        let dbg = make_simple_debug_info();
        let line = dbg.emit_debug_line();

        // DW_LNE_end_sequence is the last extended opcode: 0x00, 0x01, 0x01
        let len = line.len();
        assert_eq!(line[len - 3], 0x00);
        assert_eq!(line[len - 2], 0x01);
        assert_eq!(line[len - 1], 0x01);
    }

    // --- Debug str section tests ---

    #[test]
    fn test_debug_str_contains_producer() {
        let dbg = make_simple_debug_info();
        let str_data = dbg.emit_debug_str();

        // The producer string should be in the table
        let producer_bytes = PRODUCER.as_bytes();
        let found = str_data
            .windows(producer_bytes.len())
            .any(|w| w == producer_bytes);
        assert!(found, "Producer string not found in .debug_str");
    }

    #[test]
    fn test_debug_str_contains_function_name() {
        let dbg = make_simple_debug_info();
        let str_data = dbg.emit_debug_str();

        let found = str_data.windows(3).any(|w| w == b"add");
        assert!(found, "Function name 'add' not found in .debug_str");
    }

    #[test]
    fn test_debug_str_starts_with_null() {
        let dbg = make_simple_debug_info();
        let str_data = dbg.emit_debug_str();
        assert_eq!(str_data[0], 0);
    }

    // --- LEB128 encoding tests ---

    #[test]
    fn test_uleb128_single_byte() {
        let mut buf = Vec::new();
        encode_uleb128(0, &mut buf);
        assert_eq!(buf, &[0]);

        buf.clear();
        encode_uleb128(127, &mut buf);
        assert_eq!(buf, &[127]);
    }

    #[test]
    fn test_uleb128_multi_byte() {
        let mut buf = Vec::new();
        encode_uleb128(128, &mut buf);
        assert_eq!(buf, &[0x80, 0x01]);
    }

    #[test]
    fn test_decode_uleb128() {
        let data = [0x80, 0x01, 0xFF];
        let (val, len) = decode_uleb128(&data);
        assert_eq!(val, 128);
        assert_eq!(len, 2);
    }

    // --- SourceLanguage tests ---

    #[test]
    fn test_source_language_rust() {
        assert_eq!(SourceLanguage::Rust.dwarf_encoding(), DW_LANG_RUST);
    }

    #[test]
    fn test_source_language_c99() {
        assert_eq!(SourceLanguage::C99.dwarf_encoding(), DW_LANG_C99);
    }

    // --- BaseType tests ---

    #[test]
    fn test_base_type_i32() {
        let ty = BaseType::i32();
        assert_eq!(ty.name, "i32");
        assert_eq!(ty.byte_size, 4);
        assert_eq!(ty.encoding, DW_ATE_SIGNED);
    }

    #[test]
    fn test_base_type_f64() {
        let ty = BaseType::f64();
        assert_eq!(ty.name, "f64");
        assert_eq!(ty.byte_size, 8);
        assert_eq!(ty.encoding, DW_ATE_FLOAT);
    }

    #[test]
    fn test_base_type_boolean() {
        let ty = BaseType::boolean();
        assert_eq!(ty.name, "bool");
        assert_eq!(ty.byte_size, 1);
        assert_eq!(ty.encoding, DW_ATE_BOOLEAN);
    }

    // --- Standard types ---

    #[test]
    fn test_add_standard_types() {
        let mut dbg = DwarfDebugInfo::new("test.rs", "/tmp", SourceLanguage::Rust);
        dbg.add_standard_types();
        assert_eq!(dbg.base_types.len(), 7);
        assert_eq!(dbg.base_types[0].name, "i8");
        assert_eq!(dbg.base_types[6].name, "bool");
    }

    // --- MachOWriter integration tests ---

    #[test]
    fn test_add_debug_info_to_writer_empty() {
        let dbg = DwarfDebugInfo::new("test.rs", "/tmp", SourceLanguage::Rust);
        let mut writer = MachOWriter::new();
        writer.add_text_section(&[0x1F, 0x20, 0x03, 0xD5]);
        writer.add_symbol("_main", 1, 0, true);

        let result = add_debug_info_to_writer(&mut writer, &dbg);
        assert!(result.is_none(), "Should not add sections for empty debug info");
    }

    #[test]
    fn test_add_debug_info_to_writer_with_function() {
        let dbg = make_simple_debug_info();
        let mut writer = MachOWriter::new();
        writer.add_text_section(&[0x1F, 0x20, 0x03, 0xD5, 0xC0, 0x03, 0x5F, 0xD6]);
        writer.add_symbol("_add", 1, 0, true);

        let result = add_debug_info_to_writer(&mut writer, &dbg);
        assert!(result.is_some());

        let (abbrev_idx, info_idx, str_idx, line_idx) = result.unwrap();
        // Section indices should be sequential after __text (index 0)
        assert_eq!(abbrev_idx, 1);
        assert_eq!(info_idx, 2);
        assert_eq!(str_idx, 3);
        assert_eq!(line_idx, 4);
    }

    #[test]
    fn test_macho_with_debug_sections_produces_valid_file() {
        let dbg = make_simple_debug_info();
        let mut writer = MachOWriter::new();
        // ARM64: ADD X0, X0, X1; RET
        writer.add_text_section(&[
            0x00, 0x00, 0x01, 0x8B, // ADD X0, X0, X1
            0xC0, 0x03, 0x5F, 0xD6, // RET
        ]);
        writer.add_symbol("_add", 1, 0, true);
        add_debug_info_to_writer(&mut writer, &dbg);

        let bytes = writer.write();
        // Should produce a valid Mach-O file
        assert_eq!(&bytes[0..4], &[0xCF, 0xFA, 0xED, 0xFE]);
        // File type = MH_OBJECT
        let filetype = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
        assert_eq!(filetype, 1); // MH_OBJECT
    }

    #[test]
    fn test_debug_info_with_params_produces_valid_output() {
        let mut dbg = DwarfDebugInfo::new("test.rs", "/tmp", SourceLanguage::Rust);
        dbg.add_function(FunctionDebugInfo {
            name: "add".to_string(),
            low_pc: 0,
            size: 8,
            params: vec![
                FunctionParam { name: "a".to_string(), type_index: 0 },
                FunctionParam { name: "b".to_string(), type_index: 0 },
            ],
        });
        dbg.add_base_type(BaseType::i32());

        let info = dbg.emit_debug_info();
        let unit_length = u32::from_le_bytes(info[0..4].try_into().unwrap());
        assert_eq!(unit_length as usize, info.len() - 4);
    }

    #[test]
    fn test_c99_language_debug_info() {
        let mut dbg = DwarfDebugInfo::new("main.c", "/tmp", SourceLanguage::C99);
        dbg.add_function(FunctionDebugInfo {
            name: "main".to_string(),
            low_pc: 0,
            size: 16,
            params: vec![],
        });

        let info = dbg.emit_debug_info();
        // Check that language field is C99 (0x000c) at the correct offset
        // After unit_length(4) + version(2) + abbrev_offset(4) + addr_size(1)
        // + ULEB128(1) = compile_unit abbrev code
        // + strp(4) = producer
        // Then language (data2)
        let base = 11; // CU header size
        let (_, abbrev_len) = decode_uleb128(&info[base..]);
        let lang_offset = base + abbrev_len + 4; // skip abbrev code + producer strp
        let lang = u16::from_le_bytes(
            info[lang_offset..lang_offset + 2].try_into().unwrap(),
        );
        assert_eq!(lang, DW_LANG_C99);
    }

    #[test]
    fn test_full_round_trip_with_writer() {
        let dbg = make_multi_function_debug_info();
        let mut writer = MachOWriter::new();

        // Emit some fake code (20 bytes = two functions)
        let code = vec![
            0x00, 0x00, 0x01, 0x8B, // ADD
            0xC0, 0x03, 0x5F, 0xD6, // RET
            0x00, 0x00, 0x01, 0xCB, // SUB
            0x00, 0x00, 0x01, 0xCB, // SUB
            0xC0, 0x03, 0x5F, 0xD6, // RET
        ];
        writer.add_text_section(&code);
        writer.add_symbol("_add", 1, 0, true);
        writer.add_symbol("_sub", 1, 8, true);

        let result = add_debug_info_to_writer(&mut writer, &dbg);
        assert!(result.is_some());

        let bytes = writer.write();

        // Valid Mach-O magic
        assert_eq!(&bytes[0..4], &[0xCF, 0xFA, 0xED, 0xFE]);

        // Should have 5 sections: __text + 4 debug sections
        // nsects is in the LC_SEGMENT_64 command, which follows the header
        // Header is 32 bytes, then LC_SEGMENT_64 starts
        // LC_SEGMENT_64: cmd(4) + cmdsize(4) + segname(16) + vmaddr(8) + vmsize(8)
        //   + fileoff(8) + filesize(8) + maxprot(4) + initprot(4) + nsects(4) + flags(4)
        let nsects_offset = 32 + 4 + 4 + 16 + 8 + 8 + 8 + 8 + 4 + 4;
        let nsects = u32::from_le_bytes(
            bytes[nsects_offset..nsects_offset + 4].try_into().unwrap(),
        );
        assert_eq!(nsects, 5, "Expected 5 sections: __text + 4 DWARF");
    }
}
