// llvm2-codegen - DWARF debug information emission
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0
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

/// DW_TAG_variable — local variable.
const DW_TAG_VARIABLE: u16 = 0x34;

/// DW_TAG_lexical_block — scope within a function.
const DW_TAG_LEXICAL_BLOCK: u16 = 0x0b;

/// DW_TAG_pointer_type — pointer to another type.
const DW_TAG_POINTER_TYPE: u16 = 0x0f;

/// DW_TAG_structure_type — aggregate (struct) type.
const DW_TAG_STRUCTURE_TYPE: u16 = 0x13;

/// DW_TAG_member — member of an aggregate type.
const DW_TAG_MEMBER: u16 = 0x0d;

/// DW_TAG_enumeration_type — enumeration type.
const DW_TAG_ENUMERATION_TYPE: u16 = 0x04;

/// DW_TAG_enumerator — single enum case/value.
const DW_TAG_ENUMERATOR: u16 = 0x28;

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

/// DW_AT_location — location description for a variable.
const DW_AT_LOCATION: u16 = 0x02;

/// DW_AT_frame_base — frame base expression for a subprogram.
const DW_AT_FRAME_BASE: u16 = 0x40;

/// DW_AT_decl_file — source file index where entity is declared.
const DW_AT_DECL_FILE: u16 = 0x3a;

/// DW_AT_decl_line — source line where entity is declared.
const DW_AT_DECL_LINE: u16 = 0x3b;

/// DW_AT_linkage_name — mangled/linkage name (DWARF 4).
///
/// Reserved for future mangled-name emission (tracked under #63 DWARF work);
/// kept here for completeness of the DWARF 4 attribute table.
#[allow(dead_code)] // spec-reference constant; emitted once symbol mangling lands (#63)
const DW_AT_LINKAGE_NAME: u16 = 0x6e;

/// DW_AT_data_member_location — byte offset of a struct member from the
/// containing structure's base. DWARF 4 allows a constant (data1/data2/data4)
/// or an expression (exprloc). We emit constant form (data4) for simplicity.
const DW_AT_DATA_MEMBER_LOCATION: u16 = 0x38;

/// DW_AT_const_value — constant value of an enumerator.
const DW_AT_CONST_VALUE: u16 = 0x1c;

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

/// DW_FORM_exprloc — DWARF expression (ULEB128 length + expression bytes).
const DW_FORM_EXPRLOC: u8 = 0x18;

/// DW_FORM_sdata — signed LEB128 integer.
const DW_FORM_SDATA: u8 = 0x0d;

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
// DWARF constants — Location expression operations (DW_OP_*)
// ---------------------------------------------------------------------------

/// DW_OP_reg0 through DW_OP_reg31: value is in register N.
/// DW_OP_regN = DW_OP_REG0 + N.
const DW_OP_REG0: u8 = 0x50;

/// DW_OP_breg0 through DW_OP_breg31: value at register N + SLEB128 offset.
/// DW_OP_bregN = DW_OP_BREG0 + N.
const DW_OP_BREG0: u8 = 0x70;

/// DW_OP_regx: value in register (ULEB128 register number).
/// Used for registers > 31.
#[allow(dead_code)]
const DW_OP_REGX: u8 = 0x90;

/// DW_OP_fbreg: value at frame_base + SLEB128 offset.
const DW_OP_FBREG: u8 = 0x91;

// ---------------------------------------------------------------------------
// DWARF constants — Line number program standard opcodes
// ---------------------------------------------------------------------------

/// DW_LNS_advance_line: advance line by SLEB128 operand.
const DW_LNS_ADVANCE_LINE: u8 = 3;

/// DW_LNS_set_column: set column to ULEB128 operand.
const DW_LNS_SET_COLUMN: u8 = 5;

/// DW_LNS_negate_stmt: toggle is_stmt flag.
const DW_LNS_NEGATE_STMT: u8 = 6;

/// DW_LNS_advance_pc: advance address by ULEB128 operand (in units of min_instruction_length).
const DW_LNS_ADVANCE_PC: u8 = 2;

// ---------------------------------------------------------------------------
// Line number program parameters
// ---------------------------------------------------------------------------

/// Line base for special opcode encoding.
const LINE_BASE: i8 = -5;

/// Line range for special opcode encoding.
const LINE_RANGE: u8 = 14;

/// Opcode base (standard opcodes 1-12).
const OPCODE_BASE: u8 = 13;

/// Minimum instruction length (4 bytes for AArch64).
const MIN_INST_LEN: u8 = 4;

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

/// Abbreviation ID for DW_TAG_variable.
const ABBREV_VARIABLE: u32 = 6;

/// Abbreviation ID for DW_TAG_pointer_type.
const ABBREV_POINTER_TYPE: u32 = 7;

/// Abbreviation ID for DW_TAG_lexical_block.
const ABBREV_LEXICAL_BLOCK: u32 = 8;

/// Abbreviation ID for DW_TAG_structure_type (has children — member DIEs).
const ABBREV_STRUCTURE_TYPE: u32 = 9;

/// Abbreviation ID for DW_TAG_member (no children).
const ABBREV_MEMBER: u32 = 10;

/// Abbreviation ID for DW_TAG_enumeration_type (has children).
const ABBREV_ENUMERATION_TYPE: u32 = 11;

/// Abbreviation ID for DW_TAG_enumerator (no children).
const ABBREV_ENUMERATOR: u32 = 12;

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
    /// Standard u8 type.
    pub fn u8() -> Self { Self::unsigned("u8", 1) }
    /// Standard u16 type.
    pub fn u16() -> Self { Self::unsigned("u16", 2) }
    /// Standard u32 type.
    pub fn u32() -> Self { Self::unsigned("u32", 4) }
    /// Standard u64 type.
    pub fn u64() -> Self { Self::unsigned("u64", 8) }
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
// VariableLocation — where a variable lives at runtime
// ---------------------------------------------------------------------------

/// Runtime location of a variable for DWARF location expressions.
///
/// Reference: DWARF 4 spec, Section 2.6 (Location Descriptions)
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VariableLocation {
    /// On the stack at frame_base + offset (emits DW_OP_fbreg).
    FrameOffset(i64),
    /// In a register (emits DW_OP_reg0..DW_OP_reg31 or DW_OP_regx).
    Register(u64),
    /// At a register + offset (emits DW_OP_breg0..DW_OP_breg31).
    RegOffset(u64, i64),
}

// ---------------------------------------------------------------------------
// VariableDebugInfo — per-variable debug metadata
// ---------------------------------------------------------------------------

/// Debug information for a local variable in a function.
#[derive(Debug, Clone)]
pub struct VariableDebugInfo {
    /// Variable name.
    pub name: String,
    /// Index into the type arrays (base_types first, then pointer_types).
    pub type_index: usize,
    /// Where the variable lives at runtime.
    pub location: VariableLocation,
}

// ---------------------------------------------------------------------------
// SourceLineEntry — source line mapping for line number program
// ---------------------------------------------------------------------------

/// A source line entry mapping a code address to a source location.
///
/// Used to build the DWARF line number program which maps PC addresses
/// to source file/line/column.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceLineEntry {
    /// Code address (offset within __text section).
    pub address: u64,
    /// Source line number (1-based).
    pub line: u32,
    /// Source column number (0 = unknown).
    pub column: u16,
    /// Whether this address is a recommended breakpoint (is_stmt).
    pub is_stmt: bool,
}

// ---------------------------------------------------------------------------
// LexicalBlock — scope within a function
// ---------------------------------------------------------------------------

/// A lexical block scope within a function.
///
/// Represents a source-level scope (e.g., a `{ ... }` block in Rust/C)
/// that has its own set of local variables. The debugger uses this to
/// determine which variables are in scope at any given PC address.
///
/// DWARF: DW_TAG_lexical_block with DW_AT_low_pc + DW_AT_high_pc.
#[derive(Debug, Clone)]
pub struct LexicalBlock {
    /// Start address of the scope (offset within __text section).
    pub low_pc: u64,
    /// Size of the scope in bytes.
    pub size: u32,
    /// Variables declared in this scope.
    pub variables: Vec<VariableDebugInfo>,
}

// ---------------------------------------------------------------------------
// PointerType — pointer type descriptor
// ---------------------------------------------------------------------------

/// A DWARF pointer type (DW_TAG_pointer_type) referencing another type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PointerType {
    /// Index of the pointee type in the base_types array.
    pub pointee_type_index: usize,
    /// Size of the pointer in bytes (8 for 64-bit).
    pub byte_size: u8,
}

// ---------------------------------------------------------------------------
// StructType / StructMember — aggregate (struct) type descriptors
// ---------------------------------------------------------------------------

/// A DWARF structure type (DW_TAG_structure_type).
///
/// Represents a tMIR aggregate type (e.g., Rust `struct`, C `struct`).
/// Members are emitted as `DW_TAG_member` children with a byte offset
/// (`DW_AT_data_member_location`) and a type reference (`DW_AT_type`).
///
/// # Type references
///
/// In v1, member type indices refer into the `base_types` array only.
/// Pointer/nested-struct members are a follow-up (see the issue #326 audit
/// report for the full TypeRef refactor plan).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructType {
    /// Struct name (e.g., `"Point"`).
    pub name: String,
    /// Total size of the struct in bytes (including padding).
    pub byte_size: u32,
    /// Struct members in declaration order.
    pub members: Vec<StructMember>,
}

/// A single member of a structure type (DW_TAG_member).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructMember {
    /// Member name (e.g., `"x"`).
    pub name: String,
    /// Byte offset from the start of the containing struct.
    pub offset: u32,
    /// Index into the `base_types` array for the member's type.
    pub type_index: usize,
}

// ---------------------------------------------------------------------------
// EnumType / Enumerator — enumeration type descriptors
// ---------------------------------------------------------------------------

/// A DWARF enumeration type (DW_TAG_enumeration_type).
///
/// Enumerators are emitted as `DW_TAG_enumerator` children with a signed
/// constant value (`DW_AT_const_value`) and an underlying base type reference
/// (`DW_AT_type`) on the parent enum DIE.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnumType {
    /// Enum name (e.g., `"Color"`).
    pub name: String,
    /// Storage size of the enum in bytes.
    pub byte_size: u32,
    /// Index into the `base_types` array for the enum's underlying integer type.
    pub underlying_type_index: usize,
    /// Enumerators in declaration order.
    pub enumerators: Vec<Enumerator>,
}

/// A single enumerator of an enumeration type (DW_TAG_enumerator).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Enumerator {
    /// Enumerator name (e.g., `"Red"`).
    pub name: String,
    /// Enumerator value.
    pub value: i64,
}

// ---------------------------------------------------------------------------
// FunctionDebugInfo — per-function debug metadata
// ---------------------------------------------------------------------------

/// Debug information for a single function.
#[derive(Debug, Clone)]
pub struct FunctionDebugInfo {
    /// Function name (without Mach-O '_' prefix).
    pub name: String,
    /// Linkage name (mangled symbol name). `None` if same as `name`.
    pub linkage_name: Option<String>,
    /// Start address of the function (offset within __text section).
    pub low_pc: u64,
    /// Size of the function in bytes.
    pub size: u32,
    /// Function parameters.
    pub params: Vec<FunctionParam>,
    /// Local variables (emitted as DW_TAG_variable children).
    pub variables: Vec<VariableDebugInfo>,
    /// Lexical block scopes within this function.
    pub scopes: Vec<LexicalBlock>,
    /// Source line entries for the line number program.
    pub line_entries: Vec<SourceLineEntry>,
    /// Source file index (1-based, 0 = unknown).
    pub decl_file: u8,
    /// Source line number where function is declared (0 = unknown).
    pub decl_line: u16,
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
///     "./project",
///     SourceLanguage::Rust,
/// );
/// dbg.add_function(FunctionDebugInfo {
///     name: "add".to_string(),
///     linkage_name: None,
///     low_pc: 0,
///     size: 8,
///     params: vec![],
///     variables: vec![],
///     scopes: vec![],
///     line_entries: vec![],
///     decl_file: 1,
///     decl_line: 1,
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
    /// Pointer types used by this compilation unit.
    pub pointer_types: Vec<PointerType>,
    /// Structure (aggregate) types used by this compilation unit.
    pub struct_types: Vec<StructType>,
    /// Enumeration types used by this compilation unit.
    pub enum_types: Vec<EnumType>,
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
            pointer_types: Vec::new(),
            struct_types: Vec::new(),
            enum_types: Vec::new(),
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

    /// Add a pointer type to the debug info.
    pub fn add_pointer_type(&mut self, ty: PointerType) {
        self.pointer_types.push(ty);
    }

    /// Add a struct type to the debug info.
    ///
    /// Member `type_index` values must be valid indices into `base_types`
    /// at the time `emit_debug_info()` is called. See [`StructType`] for
    /// the v1 scope (base-type members only).
    pub fn add_struct_type(&mut self, ty: StructType) {
        self.struct_types.push(ty);
    }

    /// Add an enumeration type to the debug info.
    ///
    /// `underlying_type_index` values must be valid indices into `base_types`
    /// at the time `emit_debug_info()` is called.
    pub fn add_enum_type(&mut self, ty: EnumType) {
        self.enum_types.push(ty);
    }

    /// Add standard tMIR base types (i8, i16, i32, i64, f32, f64, bool, u8, u16, u32, u64).
    ///
    /// Index mapping after this call:
    /// - 0: i8, 1: i16, 2: i32, 3: i64, 4: f32, 5: f64, 6: bool
    /// - 7: u8, 8: u16, 9: u32, 10: u64
    pub fn add_standard_types(&mut self) {
        self.base_types.push(BaseType::i8());
        self.base_types.push(BaseType::i16());
        self.base_types.push(BaseType::i32());
        self.base_types.push(BaseType::i64());
        self.base_types.push(BaseType::f32());
        self.base_types.push(BaseType::f64());
        self.base_types.push(BaseType::boolean());
        self.base_types.push(BaseType::u8());
        self.base_types.push(BaseType::u16());
        self.base_types.push(BaseType::u32());
        self.base_types.push(BaseType::u64());
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
            if let Some(ref linkage) = func.linkage_name {
                table.insert(linkage);
            }
            for param in &func.params {
                table.insert(&param.name);
            }
            for var in &func.variables {
                table.insert(&var.name);
            }
            for scope in &func.scopes {
                for var in &scope.variables {
                    table.insert(&var.name);
                }
            }
        }

        for ty in &self.base_types {
            table.insert(&ty.name);
        }

        for sty in &self.struct_types {
            table.insert(&sty.name);
            for member in &sty.members {
                table.insert(&member.name);
            }
        }

        for ety in &self.enum_types {
            table.insert(&ety.name);
            for enumerator in &ety.enumerators {
                table.insert(&enumerator.name);
            }
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

        // Abbreviation 2: DW_TAG_subprogram (has children — for functions with params/vars)
        encode_uleb128(ABBREV_SUBPROGRAM as u64, &mut data);
        encode_uleb128(DW_TAG_SUBPROGRAM as u64, &mut data);
        data.push(DW_CHILDREN_YES);
        emit_abbrev_attr(&mut data, DW_AT_NAME, DW_FORM_STRP);
        emit_abbrev_attr(&mut data, DW_AT_LOW_PC, DW_FORM_ADDR);
        emit_abbrev_attr(&mut data, DW_AT_HIGH_PC, DW_FORM_DATA4);
        emit_abbrev_attr(&mut data, DW_AT_FRAME_BASE, DW_FORM_EXPRLOC);
        emit_abbrev_attr(&mut data, DW_AT_DECL_FILE, DW_FORM_DATA1);
        emit_abbrev_attr(&mut data, DW_AT_DECL_LINE, DW_FORM_DATA2);
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

        // Abbreviation 5: DW_TAG_subprogram (leaf — no children, no params/vars)
        encode_uleb128(ABBREV_SUBPROGRAM_LEAF as u64, &mut data);
        encode_uleb128(DW_TAG_SUBPROGRAM as u64, &mut data);
        data.push(DW_CHILDREN_NO);
        emit_abbrev_attr(&mut data, DW_AT_NAME, DW_FORM_STRP);
        emit_abbrev_attr(&mut data, DW_AT_LOW_PC, DW_FORM_ADDR);
        emit_abbrev_attr(&mut data, DW_AT_HIGH_PC, DW_FORM_DATA4);
        emit_abbrev_attr(&mut data, DW_AT_FRAME_BASE, DW_FORM_EXPRLOC);
        emit_abbrev_attr(&mut data, DW_AT_DECL_FILE, DW_FORM_DATA1);
        emit_abbrev_attr(&mut data, DW_AT_DECL_LINE, DW_FORM_DATA2);
        data.push(0);
        data.push(0);

        // Abbreviation 6: DW_TAG_variable (no children)
        encode_uleb128(ABBREV_VARIABLE as u64, &mut data);
        encode_uleb128(DW_TAG_VARIABLE as u64, &mut data);
        data.push(DW_CHILDREN_NO);
        emit_abbrev_attr(&mut data, DW_AT_NAME, DW_FORM_STRP);
        emit_abbrev_attr(&mut data, DW_AT_TYPE, DW_FORM_REF4);
        emit_abbrev_attr(&mut data, DW_AT_LOCATION, DW_FORM_EXPRLOC);
        data.push(0);
        data.push(0);

        // Abbreviation 7: DW_TAG_pointer_type (no children)
        encode_uleb128(ABBREV_POINTER_TYPE as u64, &mut data);
        encode_uleb128(DW_TAG_POINTER_TYPE as u64, &mut data);
        data.push(DW_CHILDREN_NO);
        emit_abbrev_attr(&mut data, DW_AT_TYPE, DW_FORM_REF4);
        emit_abbrev_attr(&mut data, DW_AT_BYTE_SIZE, DW_FORM_DATA1);
        data.push(0);
        data.push(0);

        // Abbreviation 8: DW_TAG_lexical_block (has children — contains variables)
        encode_uleb128(ABBREV_LEXICAL_BLOCK as u64, &mut data);
        encode_uleb128(DW_TAG_LEXICAL_BLOCK as u64, &mut data);
        data.push(DW_CHILDREN_YES);
        emit_abbrev_attr(&mut data, DW_AT_LOW_PC, DW_FORM_ADDR);
        emit_abbrev_attr(&mut data, DW_AT_HIGH_PC, DW_FORM_DATA4);
        data.push(0);
        data.push(0);

        // Abbreviation 9: DW_TAG_structure_type (has children — member DIEs).
        // Attributes: name (strp), byte_size (data4).
        encode_uleb128(ABBREV_STRUCTURE_TYPE as u64, &mut data);
        encode_uleb128(DW_TAG_STRUCTURE_TYPE as u64, &mut data);
        data.push(DW_CHILDREN_YES);
        emit_abbrev_attr(&mut data, DW_AT_NAME, DW_FORM_STRP);
        emit_abbrev_attr(&mut data, DW_AT_BYTE_SIZE, DW_FORM_DATA4);
        data.push(0);
        data.push(0);

        // Abbreviation 10: DW_TAG_member (no children).
        // Attributes: name (strp), type (ref4), data_member_location (data4).
        encode_uleb128(ABBREV_MEMBER as u64, &mut data);
        encode_uleb128(DW_TAG_MEMBER as u64, &mut data);
        data.push(DW_CHILDREN_NO);
        emit_abbrev_attr(&mut data, DW_AT_NAME, DW_FORM_STRP);
        emit_abbrev_attr(&mut data, DW_AT_TYPE, DW_FORM_REF4);
        emit_abbrev_attr(&mut data, DW_AT_DATA_MEMBER_LOCATION, DW_FORM_DATA4);
        data.push(0);
        data.push(0);

        // Abbreviation 11: DW_TAG_enumeration_type (has children — enumerators).
        // Attributes: name (strp), byte_size (data4), type (ref4).
        encode_uleb128(ABBREV_ENUMERATION_TYPE as u64, &mut data);
        encode_uleb128(DW_TAG_ENUMERATION_TYPE as u64, &mut data);
        data.push(DW_CHILDREN_YES);
        emit_abbrev_attr(&mut data, DW_AT_NAME, DW_FORM_STRP);
        emit_abbrev_attr(&mut data, DW_AT_BYTE_SIZE, DW_FORM_DATA4);
        emit_abbrev_attr(&mut data, DW_AT_TYPE, DW_FORM_REF4);
        data.push(0);
        data.push(0);

        // Abbreviation 12: DW_TAG_enumerator (no children).
        // Attributes: name (strp), const_value (sdata).
        encode_uleb128(ABBREV_ENUMERATOR as u64, &mut data);
        encode_uleb128(DW_TAG_ENUMERATOR as u64, &mut data);
        data.push(DW_CHILDREN_NO);
        emit_abbrev_attr(&mut data, DW_AT_NAME, DW_FORM_STRP);
        emit_abbrev_attr(&mut data, DW_AT_CONST_VALUE, DW_FORM_SDATA);
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
            let has_children = !func.params.is_empty()
                || !func.variables.is_empty()
                || !func.scopes.is_empty();
            let abbrev = if has_children {
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

            // DW_AT_frame_base (exprloc) — AArch64: DW_OP_reg29 (FP = X29)
            let frame_base = encode_frame_base_expr();
            encode_uleb128(frame_base.len() as u64, &mut body);
            body.extend_from_slice(&frame_base);

            // DW_AT_decl_file (data1)
            body.push(func.decl_file);
            // DW_AT_decl_line (data2)
            body.extend_from_slice(&func.decl_line.to_le_bytes());

            if has_children {
                // Emit formal parameter DIEs
                for param in &func.params {
                    encode_uleb128(ABBREV_FORMAL_PARAMETER as u64, &mut body);
                    // DW_AT_name (strp)
                    body.extend_from_slice(&str_table.offset_of(&param.name).to_le_bytes());
                    // DW_AT_type (ref4) — CU-relative offset to base type DIE
                    // Placeholder, fixed up below.
                    body.extend_from_slice(&0u32.to_le_bytes());
                }

                // Emit variable DIEs
                for var in &func.variables {
                    encode_uleb128(ABBREV_VARIABLE as u64, &mut body);
                    // DW_AT_name (strp)
                    body.extend_from_slice(&str_table.offset_of(&var.name).to_le_bytes());
                    // DW_AT_type (ref4) — placeholder, fixed up below
                    body.extend_from_slice(&0u32.to_le_bytes());
                    // DW_AT_location (exprloc)
                    let loc_expr = encode_location_expr(&var.location);
                    encode_uleb128(loc_expr.len() as u64, &mut body);
                    body.extend_from_slice(&loc_expr);
                }

                // Emit lexical block scopes with their variables
                for scope in &func.scopes {
                    encode_uleb128(ABBREV_LEXICAL_BLOCK as u64, &mut body);
                    // DW_AT_low_pc (addr)
                    body.extend_from_slice(&scope.low_pc.to_le_bytes());
                    // DW_AT_high_pc (data4) — length
                    body.extend_from_slice(&scope.size.to_le_bytes());

                    // Emit scoped variable DIEs
                    for var in &scope.variables {
                        encode_uleb128(ABBREV_VARIABLE as u64, &mut body);
                        // DW_AT_name (strp)
                        body.extend_from_slice(
                            &str_table.offset_of(&var.name).to_le_bytes(),
                        );
                        // DW_AT_type (ref4) — placeholder, fixed up below
                        body.extend_from_slice(&0u32.to_le_bytes());
                        // DW_AT_location (exprloc)
                        let loc_expr = encode_location_expr(&var.location);
                        encode_uleb128(loc_expr.len() as u64, &mut body);
                        body.extend_from_slice(&loc_expr);
                    }

                    // End of lexical_block children
                    body.push(0);
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

        // Record offset where pointer types start.
        // Pointer type index = base_types.len() + i in the type_offsets array.
        let pointer_type_offsets: Vec<u32> = {
            let mut offsets = Vec::new();
            let mut offset = body.len() as u32;
            for _ty in &self.pointer_types {
                offsets.push(offset);
                // Size: uleb128(abbrev) + ref4(4) + data1(1)
                let mut tmp = Vec::new();
                encode_uleb128(ABBREV_POINTER_TYPE as u64, &mut tmp);
                offset += tmp.len() as u32 + 4 + 1;
            }
            offsets
        };

        // Emit pointer type DIEs
        for pty in &self.pointer_types {
            encode_uleb128(ABBREV_POINTER_TYPE as u64, &mut body);
            // DW_AT_type (ref4) — reference to pointee base type
            let pointee_offset = if pty.pointee_type_index < base_type_offsets.len() {
                base_type_offsets[pty.pointee_type_index]
            } else {
                0
            };
            body.extend_from_slice(&pointee_offset.to_le_bytes());
            // DW_AT_byte_size (data1)
            body.push(pty.byte_size);
        }

        // Emit structure type DIEs (DW_TAG_structure_type with DW_TAG_member
        // children). Members reference base types by index. A trailing null
        // byte terminates the member list for each struct.
        //
        // Note: we do not record struct DIE offsets in the type-reference
        // table because v1 restricts formal parameters, variables, and struct
        // members themselves to base-type references only. Extending the
        // TypeRef scheme to allow struct references is tracked separately.
        for sty in &self.struct_types {
            encode_uleb128(ABBREV_STRUCTURE_TYPE as u64, &mut body);
            // DW_AT_name (strp)
            body.extend_from_slice(&str_table.offset_of(&sty.name).to_le_bytes());
            // DW_AT_byte_size (data4)
            body.extend_from_slice(&sty.byte_size.to_le_bytes());

            // Emit member DIEs
            for member in &sty.members {
                encode_uleb128(ABBREV_MEMBER as u64, &mut body);
                // DW_AT_name (strp)
                body.extend_from_slice(
                    &str_table.offset_of(&member.name).to_le_bytes(),
                );
                // DW_AT_type (ref4) — reference to base type
                let type_offset = if member.type_index < base_type_offsets.len() {
                    base_type_offsets[member.type_index]
                } else {
                    0
                };
                body.extend_from_slice(&type_offset.to_le_bytes());
                // DW_AT_data_member_location (data4) — byte offset
                body.extend_from_slice(&member.offset.to_le_bytes());
            }

            // End of structure_type children
            body.push(0);
        }

        // Emit enumeration type DIEs (DW_TAG_enumeration_type with
        // DW_TAG_enumerator children). The enum's underlying integer type
        // references a base type by index. A trailing null byte terminates
        // the enumerator list for each enum.
        for ety in &self.enum_types {
            encode_uleb128(ABBREV_ENUMERATION_TYPE as u64, &mut body);
            // DW_AT_name (strp)
            body.extend_from_slice(&str_table.offset_of(&ety.name).to_le_bytes());
            // DW_AT_byte_size (data4)
            body.extend_from_slice(&ety.byte_size.to_le_bytes());
            // DW_AT_type (ref4) — reference to underlying base type
            let underlying_type_offset =
                if ety.underlying_type_index < base_type_offsets.len() {
                    base_type_offsets[ety.underlying_type_index]
                } else {
                    0
                };
            body.extend_from_slice(&underlying_type_offset.to_le_bytes());

            // Emit enumerator DIEs
            for enumerator in &ety.enumerators {
                encode_uleb128(ABBREV_ENUMERATOR as u64, &mut body);
                // DW_AT_name (strp)
                body.extend_from_slice(
                    &str_table.offset_of(&enumerator.name).to_le_bytes(),
                );
                // DW_AT_const_value (sdata)
                encode_sleb128(enumerator.value, &mut body);
            }

            // End of enumeration_type children
            body.push(0);
        }

        // End of compile_unit children
        body.push(0);

        // Build the combined type offset table: base types first, then pointer types.
        let all_type_offsets: Vec<u32> = base_type_offsets
            .iter()
            .chain(pointer_type_offsets.iter())
            .copied()
            .collect();

        // Now fix up the DW_AT_type references in formal_parameter and variable DIEs.
        if !all_type_offsets.is_empty() {
            self.fixup_type_refs(&mut body, &str_table, &all_type_offsets);
        }

        // Prepend unit_length (4 bytes, length of everything after this field)
        let unit_length = body.len() as u32;
        let mut result = Vec::with_capacity(4 + body.len());
        result.extend_from_slice(&unit_length.to_le_bytes());
        result.extend(body);

        result
    }

    /// Fix up DW_AT_type references in formal parameter and variable DIEs.
    ///
    /// Walks the body bytes looking for formal parameter and variable DIEs
    /// and patches their type reference (DW_FORM_ref4) to point to the
    /// correct type DIE (base type or pointer type).
    fn fixup_type_refs(
        &self,
        body: &mut [u8],
        _str_table: &StringTable,
        type_offsets: &[u32],
    ) {
        // Skip CU header: version(2) + abbrev_offset(4) + addr_size(1) = 7
        let mut offset = 7;

        // Skip compile_unit DIE abbreviation code
        let (_, len) = decode_uleb128(&body[offset..]);
        offset += len;

        // Skip compile_unit attributes:
        // producer(4) + language(2) + name(4) + comp_dir(4) + low_pc(8) + high_pc(4) + stmt_list(4)
        offset += 4 + 2 + 4 + 4 + 8 + 4 + 4; // = 30

        // Now we're at the subprogram DIEs
        for func in &self.functions {
            let has_children = !func.params.is_empty()
                || !func.variables.is_empty()
                || !func.scopes.is_empty();

            // Skip subprogram abbreviation code
            let (_, len) = decode_uleb128(&body[offset..]);
            offset += len;

            // Skip subprogram attributes: name(4) + low_pc(8) + high_pc(4) = 16
            offset += 4 + 8 + 4;

            // Skip frame_base (exprloc): ULEB128 length + expression bytes
            let (expr_len, len) = decode_uleb128(&body[offset..]);
            offset += len + expr_len as usize;

            // Skip decl_file (data1) + decl_line (data2) = 3
            offset += 1 + 2;

            if has_children {
                // Fix up formal parameter type references
                for param in &func.params {
                    // Skip formal_parameter abbreviation code
                    let (_, len) = decode_uleb128(&body[offset..]);
                    offset += len;

                    // Skip name (strp, 4 bytes)
                    offset += 4;

                    // Patch type reference (ref4, 4 bytes)
                    let type_idx = param.type_index;
                    if type_idx < type_offsets.len() {
                        let type_offset = type_offsets[type_idx];
                        body[offset..offset + 4]
                            .copy_from_slice(&type_offset.to_le_bytes());
                    }
                    offset += 4;
                }

                // Fix up variable type references
                for var in &func.variables {
                    // Skip variable abbreviation code
                    let (_, len) = decode_uleb128(&body[offset..]);
                    offset += len;

                    // Skip name (strp, 4 bytes)
                    offset += 4;

                    // Patch type reference (ref4, 4 bytes)
                    let type_idx = var.type_index;
                    if type_idx < type_offsets.len() {
                        let type_offset = type_offsets[type_idx];
                        body[offset..offset + 4]
                            .copy_from_slice(&type_offset.to_le_bytes());
                    }
                    offset += 4;

                    // Skip location (exprloc): ULEB128 length + expression bytes
                    let (expr_len, len) = decode_uleb128(&body[offset..]);
                    offset += len + expr_len as usize;
                }

                // Fix up variables inside lexical block scopes
                for scope in &func.scopes {
                    // Skip lexical_block abbreviation code
                    let (_, len) = decode_uleb128(&body[offset..]);
                    offset += len;

                    // Skip low_pc (addr, 8 bytes) + high_pc (data4, 4 bytes)
                    offset += 8 + 4;

                    // Fix up scoped variable type references
                    for var in &scope.variables {
                        // Skip variable abbreviation code
                        let (_, len) = decode_uleb128(&body[offset..]);
                        offset += len;

                        // Skip name (strp, 4 bytes)
                        offset += 4;

                        // Patch type reference (ref4, 4 bytes)
                        let type_idx = var.type_index;
                        if type_idx < type_offsets.len() {
                            let type_offset = type_offsets[type_idx];
                            body[offset..offset + 4]
                                .copy_from_slice(&type_offset.to_le_bytes());
                        }
                        offset += 4;

                        // Skip location (exprloc): ULEB128 length + expression bytes
                        let (expr_len, len) = decode_uleb128(&body[offset..]);
                        offset += len + expr_len as usize;
                    }

                    // Skip null terminator for lexical_block children
                    offset += 1;
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

    /// Emit the `.debug_line` section.
    ///
    /// When functions have `line_entries`, emits a full line number program
    /// with proper line/address advance using special opcodes. When functions
    /// have no line_entries, falls back to minimal function-boundary markers.
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
        header_body.push(MIN_INST_LEN);
        // maximum_operations_per_instruction (1 for non-VLIW)
        header_body.push(1);
        // default_is_stmt
        header_body.push(1);
        // line_base
        header_body.push(LINE_BASE as u8);
        // line_range
        header_body.push(LINE_RANGE);
        // opcode_base (13 = standard opcodes 1-12)
        header_body.push(OPCODE_BASE);
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
        // Set file to index 1
        header_body.push(4); // DW_LNS_set_file
        encode_uleb128(1, &mut header_body);

        // Collect all line entries from all functions, or fall back to
        // function-boundary markers.
        let has_line_entries = self.functions.iter().any(|f| !f.line_entries.is_empty());

        if has_line_entries {
            // Emit detailed line program from SourceLineEntry data.
            // State machine initial state: address=0, line=1, column=0, is_stmt=true
            #[allow(unused_assignments)]
            let mut cur_addr: u64 = 0;
            let mut cur_line: i64 = 1;
            let mut cur_col: u16 = 0;
            let mut cur_is_stmt = true;

            for func in &self.functions {
                if func.line_entries.is_empty() {
                    // Fall back to function-boundary for this function
                    emit_set_address(&mut header_body, func.low_pc);
                    header_body.push(1); // DW_LNS_copy

                    emit_set_address(&mut header_body, func.low_pc + func.size as u64);
                    continue;
                }

                // Set address to function start
                emit_set_address(&mut header_body, func.low_pc);
                cur_addr = func.low_pc;

                for entry in &func.line_entries {
                    let addr_advance = (entry.address - cur_addr) / MIN_INST_LEN as u64;
                    let line_advance = entry.line as i64 - cur_line;

                    // Toggle is_stmt if needed
                    if entry.is_stmt != cur_is_stmt {
                        header_body.push(DW_LNS_NEGATE_STMT);
                        cur_is_stmt = entry.is_stmt;
                    }

                    // Set column if changed
                    if entry.column != cur_col {
                        header_body.push(DW_LNS_SET_COLUMN);
                        encode_uleb128(entry.column as u64, &mut header_body);
                        cur_col = entry.column;
                    }

                    // Try to use a special opcode for combined line+addr advance
                    let adj_line = line_advance - LINE_BASE as i64;
                    if adj_line >= 0
                        && adj_line < LINE_RANGE as i64
                        && addr_advance < ((255 - OPCODE_BASE as u64 - adj_line as u64)
                            / LINE_RANGE as u64
                            + 1)
                    {
                        let opcode = (adj_line + LINE_RANGE as i64 * addr_advance as i64
                            + OPCODE_BASE as i64) as u8;
                        header_body.push(opcode);
                    } else {
                        // Use standard opcodes for large advances
                        if line_advance != 0 {
                            header_body.push(DW_LNS_ADVANCE_LINE);
                            encode_sleb128(line_advance, &mut header_body);
                        }
                        if addr_advance > 0 {
                            header_body.push(DW_LNS_ADVANCE_PC);
                            encode_uleb128(addr_advance, &mut header_body);
                        }
                        header_body.push(1); // DW_LNS_copy
                    }

                    cur_addr = entry.address;
                    cur_line = entry.line as i64;
                }

                // Advance to function end if needed
                let end_pc = func.low_pc + func.size as u64;
                if cur_addr < end_pc {
                    emit_set_address(&mut header_body, end_pc);
                }
            }
        } else {
            // Minimal: for each function, set address and emit a row.
            for func in &self.functions {
                emit_set_address(&mut header_body, func.low_pc);
                header_body.push(1); // DW_LNS_copy

                emit_set_address(&mut header_body, func.low_pc + func.size as u64);
            }
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
// DWARF location expression helpers
// ---------------------------------------------------------------------------

/// Encode the frame base expression for AArch64.
///
/// Returns DW_OP_reg29 (0x6D) -- the frame pointer register X29.
/// This tells the debugger that the frame base is in register X29.
pub fn encode_frame_base_expr() -> Vec<u8> {
    vec![DW_OP_REG0 + 29] // DW_OP_reg29 = 0x50 + 29 = 0x6D
}

/// Encode a DWARF location expression for a variable.
///
/// Produces the raw bytes of the expression (not including the ULEB128
/// length prefix used by DW_FORM_exprloc -- the caller handles that).
pub fn encode_location_expr(loc: &VariableLocation) -> Vec<u8> {
    let mut expr = Vec::new();
    match loc {
        VariableLocation::FrameOffset(offset) => {
            // DW_OP_fbreg(SLEB128_offset): value at frame_base + offset
            expr.push(DW_OP_FBREG);
            encode_sleb128(*offset, &mut expr);
        }
        VariableLocation::Register(reg) => {
            if *reg <= 31 {
                // DW_OP_reg0..DW_OP_reg31
                expr.push(DW_OP_REG0 + *reg as u8);
            } else {
                // DW_OP_regx for registers > 31
                expr.push(DW_OP_REGX);
                encode_uleb128(*reg, &mut expr);
            }
        }
        VariableLocation::RegOffset(reg, offset) => {
            if *reg <= 31 {
                // DW_OP_breg0..DW_OP_breg31(SLEB128_offset)
                expr.push(DW_OP_BREG0 + *reg as u8);
            } else {
                // For registers > 31, use DW_OP_bregx (0x92)
                expr.push(0x92); // DW_OP_bregx
                encode_uleb128(*reg, &mut expr);
            }
            encode_sleb128(*offset, &mut expr);
        }
    }
    expr
}

// ---------------------------------------------------------------------------
// Line program emission helper
// ---------------------------------------------------------------------------

/// Emit a DW_LNE_set_address extended opcode.
fn emit_set_address(out: &mut Vec<u8>, addr: u64) {
    out.push(0); // extended opcode marker
    encode_uleb128(1 + ADDRESS_SIZE as u64, out); // length
    out.push(0x02); // DW_LNE_set_address
    out.extend_from_slice(&addr.to_le_bytes());
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

/// Encode a value as SLEB128 (signed LEB128).
fn encode_sleb128(mut value: i64, out: &mut Vec<u8>) {
    let mut more = true;
    while more {
        let mut byte = (value & 0x7F) as u8;
        value >>= 7;
        if (value == 0 && byte & 0x40 == 0) || (value == -1 && byte & 0x40 != 0) {
            more = false;
        } else {
            byte |= 0x80;
        }
        out.push(byte);
    }
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
        let mut dbg = DwarfDebugInfo::new("test.rs", "./project", SourceLanguage::Rust);
        dbg.add_function(FunctionDebugInfo {
            name: "add".to_string(),
            linkage_name: None,
            low_pc: 0,
            size: 8,
            params: vec![],
            variables: vec![],
            scopes: vec![],
            line_entries: vec![],
            decl_file: 1,
            decl_line: 1,
        });
        dbg.add_base_type(BaseType::i32());
        dbg
    }

    fn make_multi_function_debug_info() -> DwarfDebugInfo {
        let mut dbg = DwarfDebugInfo::new("lib.rs", "./project", SourceLanguage::Rust);
        dbg.add_function(FunctionDebugInfo {
            name: "add".to_string(),
            linkage_name: None,
            low_pc: 0,
            size: 8,
            params: vec![
                FunctionParam { name: "a".to_string(), type_index: 0 },
                FunctionParam { name: "b".to_string(), type_index: 0 },
            ],
            variables: vec![],
            scopes: vec![],
            line_entries: vec![],
            decl_file: 1,
            decl_line: 10,
        });
        dbg.add_function(FunctionDebugInfo {
            name: "sub".to_string(),
            linkage_name: None,
            low_pc: 8,
            size: 12,
            params: vec![],
            variables: vec![],
            scopes: vec![],
            line_entries: vec![],
            decl_file: 1,
            decl_line: 20,
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
        assert_eq!(dbg.base_types.len(), 11);
        assert_eq!(dbg.base_types[0].name, "i8");
        assert_eq!(dbg.base_types[6].name, "bool");
        assert_eq!(dbg.base_types[7].name, "u8");
        assert_eq!(dbg.base_types[8].name, "u16");
        assert_eq!(dbg.base_types[9].name, "u32");
        assert_eq!(dbg.base_types[10].name, "u64");
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
            linkage_name: None,
            low_pc: 0,
            size: 8,
            params: vec![
                FunctionParam { name: "a".to_string(), type_index: 0 },
                FunctionParam { name: "b".to_string(), type_index: 0 },
            ],
            variables: vec![],
            scopes: vec![],
            line_entries: vec![],
            decl_file: 1,
            decl_line: 5,
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
            linkage_name: None,
            low_pc: 0,
            size: 16,
            params: vec![],
            variables: vec![],
            scopes: vec![],
            line_entries: vec![],
            decl_file: 1,
            decl_line: 1,
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

    // --- Frame base and location expression tests ---

    #[test]
    fn test_encode_frame_base_expr() {
        assert_eq!(encode_frame_base_expr(), vec![0x6D]);
    }

    #[test]
    fn test_encode_location_expr_frame_offset_zero() {
        assert_eq!(
            encode_location_expr(&VariableLocation::FrameOffset(0)),
            vec![0x91, 0x00],
        );
    }

    #[test]
    fn test_encode_location_expr_frame_offset_negative() {
        assert_eq!(
            encode_location_expr(&VariableLocation::FrameOffset(-8)),
            vec![0x91, 0x78],
        );
    }

    #[test]
    fn test_encode_location_expr_register_small() {
        assert_eq!(
            encode_location_expr(&VariableLocation::Register(0)),
            vec![0x50],
        );
        assert_eq!(
            encode_location_expr(&VariableLocation::Register(19)),
            vec![0x63],
        );
    }

    #[test]
    fn test_encode_location_expr_register_large() {
        assert_eq!(
            encode_location_expr(&VariableLocation::Register(33)),
            vec![0x90, 0x21],
        );
    }

    #[test]
    fn test_encode_location_expr_reg_offset() {
        assert_eq!(
            encode_location_expr(&VariableLocation::RegOffset(29, -16)),
            vec![0x8D, 0x70],
        );
    }

    // --- SLEB128 encoding tests ---

    #[test]
    fn test_encode_sleb128_positive() {
        let cases: [(i64, Vec<u8>); 6] = [
            (0, vec![0x00]),
            (1, vec![0x01]),
            (63, vec![0x3F]),
            (64, vec![0xC0, 0x00]),
            (127, vec![0xFF, 0x00]),
            (128, vec![0x80, 0x01]),
        ];
        for (value, expected) in &cases {
            let mut buf = Vec::new();
            encode_sleb128(*value, &mut buf);
            assert_eq!(buf, *expected, "SLEB128({}) failed", value);
        }
    }

    #[test]
    fn test_encode_sleb128_negative() {
        let cases: [(i64, Vec<u8>); 4] = [
            (-1, vec![0x7F]),
            (-64, vec![0x40]),
            (-65, vec![0xBF, 0x7F]),
            (-128, vec![0x80, 0x7F]),
        ];
        for (value, expected) in &cases {
            let mut buf = Vec::new();
            encode_sleb128(*value, &mut buf);
            assert_eq!(buf, *expected, "SLEB128({}) failed", value);
        }
    }

    // --- Variable DIE emission tests ---

    #[test]
    fn test_variable_die_emission() {
        let mut without_vars =
            DwarfDebugInfo::new("test.rs", "/tmp", SourceLanguage::Rust);
        without_vars.add_function(FunctionDebugInfo {
            name: "add".to_string(),
            linkage_name: None,
            low_pc: 0,
            size: 8,
            params: vec![],
            variables: vec![],
            scopes: vec![],
            line_entries: vec![],
            decl_file: 1,
            decl_line: 1,
        });
        without_vars.add_base_type(BaseType::i32());

        let mut with_vars =
            DwarfDebugInfo::new("test.rs", "/tmp", SourceLanguage::Rust);
        with_vars.add_function(FunctionDebugInfo {
            name: "add".to_string(),
            linkage_name: None,
            low_pc: 0,
            size: 8,
            params: vec![],
            variables: vec![VariableDebugInfo {
                name: "x".to_string(),
                type_index: 0,
                location: VariableLocation::FrameOffset(-16),
            }],
            scopes: vec![],
            line_entries: vec![],
            decl_file: 1,
            decl_line: 1,
        });
        with_vars.add_base_type(BaseType::i32());

        let info_without = without_vars.emit_debug_info();
        let info_with = with_vars.emit_debug_info();
        assert!(!info_with.is_empty());
        assert!(
            info_with.len() > info_without.len(),
            "variable DIE should increase debug_info size"
        );
    }

    // --- Pointer type DIE emission tests ---

    #[test]
    fn test_pointer_type_die_emission() {
        let mut without_ptr =
            DwarfDebugInfo::new("test.rs", "/tmp", SourceLanguage::Rust);
        without_ptr.add_function(FunctionDebugInfo {
            name: "add".to_string(),
            linkage_name: None,
            low_pc: 0,
            size: 8,
            params: vec![],
            variables: vec![],
            scopes: vec![],
            line_entries: vec![],
            decl_file: 1,
            decl_line: 1,
        });
        without_ptr.add_base_type(BaseType::i32());

        let mut with_ptr =
            DwarfDebugInfo::new("test.rs", "/tmp", SourceLanguage::Rust);
        with_ptr.add_function(FunctionDebugInfo {
            name: "add".to_string(),
            linkage_name: None,
            low_pc: 0,
            size: 8,
            params: vec![],
            variables: vec![],
            scopes: vec![],
            line_entries: vec![],
            decl_file: 1,
            decl_line: 1,
        });
        with_ptr.add_base_type(BaseType::i32());
        with_ptr.add_pointer_type(PointerType {
            pointee_type_index: 0,
            byte_size: 8,
        });

        let info_without = without_ptr.emit_debug_info();
        let info_with = with_ptr.emit_debug_info();
        assert!(!info_with.is_empty());
        assert!(
            info_with.len() > info_without.len(),
            "pointer type DIE should increase debug_info size"
        );
        // The pointer type abbreviation code (7) should appear in the output
        // after the base type entries.
        let has_ptr_abbrev = info_with.windows(1).any(|w| w[0] == ABBREV_POINTER_TYPE as u8);
        assert!(has_ptr_abbrev, "pointer type abbrev code should appear in debug_info");
    }

    // --- Line number program tests ---

    #[test]
    fn test_debug_line_with_source_entries() {
        let mut without_entries =
            DwarfDebugInfo::new("test.rs", "/tmp", SourceLanguage::Rust);
        without_entries.add_function(FunctionDebugInfo {
            name: "add".to_string(),
            linkage_name: None,
            low_pc: 0,
            size: 8,
            params: vec![],
            variables: vec![],
            scopes: vec![],
            line_entries: vec![],
            decl_file: 1,
            decl_line: 1,
        });

        let mut with_entries =
            DwarfDebugInfo::new("test.rs", "/tmp", SourceLanguage::Rust);
        with_entries.add_function(FunctionDebugInfo {
            name: "add".to_string(),
            linkage_name: None,
            low_pc: 0,
            size: 8,
            params: vec![],
            variables: vec![],
            scopes: vec![],
            line_entries: vec![
                SourceLineEntry {
                    address: 0,
                    line: 10,
                    column: 5,
                    is_stmt: true,
                },
                SourceLineEntry {
                    address: 4,
                    line: 11,
                    column: 0,
                    is_stmt: true,
                },
            ],
            decl_file: 1,
            decl_line: 1,
        });

        let line_without = without_entries.emit_debug_line();
        let line_with = with_entries.emit_debug_line();
        assert!(
            line_with.len() > line_without.len(),
            "source line entries should increase debug_line size"
        );
    }

    // --- String table variable name test ---

    #[test]
    fn test_debug_str_includes_variable_names() {
        let mut dbg = DwarfDebugInfo::new("test.rs", "/tmp", SourceLanguage::Rust);
        dbg.add_function(FunctionDebugInfo {
            name: "add".to_string(),
            linkage_name: None,
            low_pc: 0,
            size: 8,
            params: vec![],
            variables: vec![VariableDebugInfo {
                name: "count".to_string(),
                type_index: 0,
                location: VariableLocation::FrameOffset(-16),
            }],
            scopes: vec![],
            line_entries: vec![],
            decl_file: 1,
            decl_line: 1,
        });
        dbg.add_base_type(BaseType::i32());

        let str_data = dbg.emit_debug_str();
        assert!(
            str_data.windows(b"count".len()).any(|w| w == b"count"),
            "string table should contain variable name 'count'"
        );
    }

    // --- Abbreviation table new entry tests ---

    #[test]
    fn test_abbrev_includes_variable_entry() {
        let dbg = make_simple_debug_info();
        let abbrev = dbg.emit_abbrev();
        // ABBREV_VARIABLE=6, DW_TAG_VARIABLE=0x34, DW_CHILDREN_NO=0
        assert!(
            abbrev.windows(3).any(|w| {
                w == [ABBREV_VARIABLE as u8, DW_TAG_VARIABLE as u8, DW_CHILDREN_NO]
            }),
            "abbreviation table should contain variable entry"
        );
    }

    #[test]
    fn test_abbrev_includes_pointer_type_entry() {
        let dbg = make_simple_debug_info();
        let abbrev = dbg.emit_abbrev();
        // ABBREV_POINTER_TYPE=7, DW_TAG_POINTER_TYPE=0x0f, DW_CHILDREN_NO=0
        assert!(
            abbrev.windows(3).any(|w| {
                w == [ABBREV_POINTER_TYPE as u8, DW_TAG_POINTER_TYPE as u8, DW_CHILDREN_NO]
            }),
            "abbreviation table should contain pointer type entry"
        );
    }

    // --- tMIR debug metadata wiring tests ---
    //
    // These tests verify the end-to-end path from FunctionDebugMeta / SourceLoc
    // on MachFunction/MachInst through to DWARF section emission.

    #[test]
    fn test_debug_meta_param_names_appear_in_debug_str() {
        // Verify that parameter names from FunctionDebugMeta flow into
        // the DWARF string table via FunctionDebugInfo.params.
        let mut dbg = DwarfDebugInfo::new("main.rs", "/src", SourceLanguage::Rust);
        dbg.add_standard_types();
        dbg.add_function(FunctionDebugInfo {
            name: "compute".to_string(),
            linkage_name: None,
            low_pc: 0,
            size: 16,
            params: vec![
                FunctionParam { name: "arg0".to_string(), type_index: 3 }, // i64
                FunctionParam { name: "arg1".to_string(), type_index: 2 }, // i32
            ],
            variables: vec![],
            scopes: vec![],
            line_entries: vec![],
            decl_file: 1,
            decl_line: 42,
        });

        let str_data = dbg.emit_debug_str();
        assert!(
            str_data.windows(b"arg0".len()).any(|w| w == b"arg0"),
            "string table should contain param name 'arg0'"
        );
        assert!(
            str_data.windows(b"arg1".len()).any(|w| w == b"arg1"),
            "string table should contain param name 'arg1'"
        );
        assert!(
            str_data.windows(b"compute".len()).any(|w| w == b"compute"),
            "string table should contain function name 'compute'"
        );
    }

    #[test]
    fn test_source_loc_line_entries_increase_debug_line_size() {
        // Simulate the pipeline's conversion of MachInst.source_loc
        // into SourceLineEntry for DWARF line number program emission.
        let mut without_lines = DwarfDebugInfo::new("main.rs", "/src", SourceLanguage::Rust);
        without_lines.add_function(FunctionDebugInfo {
            name: "foo".to_string(),
            linkage_name: None,
            low_pc: 0,
            size: 12,
            params: vec![],
            variables: vec![],
            scopes: vec![],
            line_entries: vec![],
            decl_file: 1,
            decl_line: 1,
        });

        let mut with_lines = DwarfDebugInfo::new("main.rs", "/src", SourceLanguage::Rust);
        with_lines.add_function(FunctionDebugInfo {
            name: "foo".to_string(),
            linkage_name: None,
            low_pc: 0,
            size: 12,
            params: vec![],
            variables: vec![],
            scopes: vec![],
            line_entries: vec![
                SourceLineEntry { address: 0, line: 10, column: 1, is_stmt: true },
                SourceLineEntry { address: 4, line: 11, column: 5, is_stmt: true },
                SourceLineEntry { address: 8, line: 12, column: 0, is_stmt: true },
            ],
            decl_file: 1,
            decl_line: 10,
        });

        let line_without = without_lines.emit_debug_line();
        let line_with = with_lines.emit_debug_line();
        assert!(
            line_with.len() > line_without.len(),
            "source line entries from MachInst source_loc should increase debug_line size"
        );
    }

    #[test]
    fn test_stack_slot_variable_debug_info_emission() {
        // Verify that stack slot-derived VariableDebugInfo (as built by
        // emit_macho) produces valid DWARF debug_info with variable DIEs.
        let mut dbg = DwarfDebugInfo::new("lib.rs", "/src", SourceLanguage::Rust);
        dbg.add_standard_types();
        dbg.add_function(FunctionDebugInfo {
            name: "process".to_string(),
            linkage_name: None,
            low_pc: 0,
            size: 20,
            params: vec![
                FunctionParam { name: "arg0".to_string(), type_index: 3 },
            ],
            variables: vec![
                VariableDebugInfo {
                    name: "local_0".to_string(),
                    type_index: 3, // i64 (8-byte slot)
                    location: VariableLocation::FrameOffset(-8),
                },
                VariableDebugInfo {
                    name: "local_1".to_string(),
                    type_index: 2, // i32 (4-byte slot)
                    location: VariableLocation::FrameOffset(-12),
                },
            ],
            scopes: vec![],
            line_entries: vec![
                SourceLineEntry { address: 0, line: 5, column: 0, is_stmt: true },
                SourceLineEntry { address: 8, line: 7, column: 0, is_stmt: true },
            ],
            decl_file: 1,
            decl_line: 5,
        });

        let info = dbg.emit_debug_info();
        let unit_length = u32::from_le_bytes(info[0..4].try_into().unwrap());
        assert_eq!(
            unit_length as usize,
            info.len() - 4,
            "unit_length should match actual debug_info size"
        );

        // With params + variables + line entries, the debug_info should be
        // significantly larger than a bare function (make_simple_debug_info).
        let bare_dbg = make_simple_debug_info();
        let bare_info = bare_dbg.emit_debug_info();
        assert!(
            info.len() > bare_info.len(),
            "function with params, variables, and line entries should produce larger debug_info"
        );
    }

    #[test]
    fn test_full_wiring_round_trip_with_debug_meta() {
        // End-to-end test: construct DWARF info as the pipeline does (from
        // debug_meta + source_locs) and verify the Mach-O output is valid.
        let mut dbg = DwarfDebugInfo::new("example.rs", "./project", SourceLanguage::Rust);
        dbg.add_standard_types();
        dbg.add_function(FunctionDebugInfo {
            name: "sum".to_string(),
            linkage_name: None,
            low_pc: 0,
            size: 16,
            params: vec![
                FunctionParam { name: "arg0".to_string(), type_index: 3 },
                FunctionParam { name: "arg1".to_string(), type_index: 3 },
            ],
            variables: vec![
                VariableDebugInfo {
                    name: "local_0".to_string(),
                    type_index: 3,
                    location: VariableLocation::FrameOffset(-8),
                },
            ],
            scopes: vec![],
            line_entries: vec![
                SourceLineEntry { address: 0, line: 1, column: 0, is_stmt: true },
                SourceLineEntry { address: 4, line: 2, column: 0, is_stmt: true },
                SourceLineEntry { address: 8, line: 3, column: 0, is_stmt: true },
                SourceLineEntry { address: 12, line: 3, column: 5, is_stmt: true },
            ],
            decl_file: 1,
            decl_line: 1,
        });

        let mut writer = MachOWriter::new();
        writer.add_text_section(&[
            0x00, 0x00, 0x01, 0x8B, // ADD X0, X0, X1
            0x00, 0x00, 0x01, 0x8B, // ADD X0, X0, X1
            0x00, 0x00, 0x01, 0x8B, // ADD X0, X0, X1
            0xC0, 0x03, 0x5F, 0xD6, // RET
        ]);
        writer.add_symbol("_sum", 1, 0, true);

        let result = add_debug_info_to_writer(&mut writer, &dbg);
        assert!(result.is_some(), "should add DWARF sections");

        let bytes = writer.write();
        // Valid Mach-O magic
        assert_eq!(&bytes[0..4], &[0xCF, 0xFA, 0xED, 0xFE]);

        // Should have 5 sections: __text + 4 DWARF sections
        let nsects_offset = 32 + 4 + 4 + 16 + 8 + 8 + 8 + 8 + 4 + 4;
        let nsects = u32::from_le_bytes(
            bytes[nsects_offset..nsects_offset + 4].try_into().unwrap(),
        );
        assert_eq!(nsects, 5, "Expected 5 sections: __text + 4 DWARF");
    }

    #[test]
    fn test_decl_line_propagates_to_debug_info() {
        // The decl_line from FunctionDebugMeta should appear in the
        // subprogram DIE's DW_AT_decl_line attribute.
        let mut dbg = DwarfDebugInfo::new("test.rs", "/tmp", SourceLanguage::Rust);
        dbg.add_function(FunctionDebugInfo {
            name: "my_func".to_string(),
            linkage_name: None,
            low_pc: 0,
            size: 8,
            params: vec![],
            variables: vec![],
            scopes: vec![],
            line_entries: vec![],
            decl_file: 1,
            decl_line: 99,
        });
        dbg.add_base_type(BaseType::i32());

        let info = dbg.emit_debug_info();
        // decl_line=99 should appear encoded as data2 (little-endian u16) somewhere.
        let target_le = 99u16.to_le_bytes();
        let found = info.windows(2).any(|w| w == target_le);
        assert!(found, "decl_line 99 should appear in debug_info bytes");
    }

    // --- Unsigned type tests ---

    #[test]
    fn test_base_type_u8() {
        let ty = BaseType::u8();
        assert_eq!(ty.name, "u8");
        assert_eq!(ty.byte_size, 1);
        assert_eq!(ty.encoding, DW_ATE_UNSIGNED);
    }

    #[test]
    fn test_base_type_u16() {
        let ty = BaseType::u16();
        assert_eq!(ty.name, "u16");
        assert_eq!(ty.byte_size, 2);
        assert_eq!(ty.encoding, DW_ATE_UNSIGNED);
    }

    #[test]
    fn test_base_type_u32() {
        let ty = BaseType::u32();
        assert_eq!(ty.name, "u32");
        assert_eq!(ty.byte_size, 4);
        assert_eq!(ty.encoding, DW_ATE_UNSIGNED);
    }

    #[test]
    fn test_base_type_u64() {
        let ty = BaseType::u64();
        assert_eq!(ty.name, "u64");
        assert_eq!(ty.byte_size, 8);
        assert_eq!(ty.encoding, DW_ATE_UNSIGNED);
    }

    // --- Lexical block scope tests ---

    #[test]
    fn test_lexical_block_increases_debug_info_size() {
        let mut without_scope =
            DwarfDebugInfo::new("test.rs", "/tmp", SourceLanguage::Rust);
        without_scope.add_function(FunctionDebugInfo {
            name: "foo".to_string(),
            linkage_name: None,
            low_pc: 0,
            size: 20,
            params: vec![],
            variables: vec![VariableDebugInfo {
                name: "x".to_string(),
                type_index: 0,
                location: VariableLocation::FrameOffset(-8),
            }],
            scopes: vec![],
            line_entries: vec![],
            decl_file: 1,
            decl_line: 1,
        });
        without_scope.add_base_type(BaseType::i32());

        let mut with_scope =
            DwarfDebugInfo::new("test.rs", "/tmp", SourceLanguage::Rust);
        with_scope.add_function(FunctionDebugInfo {
            name: "foo".to_string(),
            linkage_name: None,
            low_pc: 0,
            size: 20,
            params: vec![],
            variables: vec![VariableDebugInfo {
                name: "x".to_string(),
                type_index: 0,
                location: VariableLocation::FrameOffset(-8),
            }],
            scopes: vec![LexicalBlock {
                low_pc: 4,
                size: 12,
                variables: vec![VariableDebugInfo {
                    name: "y".to_string(),
                    type_index: 0,
                    location: VariableLocation::FrameOffset(-16),
                }],
            }],
            line_entries: vec![],
            decl_file: 1,
            decl_line: 1,
        });
        with_scope.add_base_type(BaseType::i32());

        let info_without = without_scope.emit_debug_info();
        let info_with = with_scope.emit_debug_info();
        assert!(
            info_with.len() > info_without.len(),
            "lexical block scope should increase debug_info size"
        );
    }

    #[test]
    fn test_lexical_block_valid_unit_length() {
        let mut dbg = DwarfDebugInfo::new("test.rs", "/tmp", SourceLanguage::Rust);
        dbg.add_function(FunctionDebugInfo {
            name: "bar".to_string(),
            linkage_name: None,
            low_pc: 0,
            size: 32,
            params: vec![],
            variables: vec![],
            scopes: vec![
                LexicalBlock {
                    low_pc: 0,
                    size: 16,
                    variables: vec![VariableDebugInfo {
                        name: "a".to_string(),
                        type_index: 0,
                        location: VariableLocation::FrameOffset(-8),
                    }],
                },
                LexicalBlock {
                    low_pc: 16,
                    size: 16,
                    variables: vec![VariableDebugInfo {
                        name: "b".to_string(),
                        type_index: 0,
                        location: VariableLocation::FrameOffset(-16),
                    }],
                },
            ],
            line_entries: vec![],
            decl_file: 1,
            decl_line: 1,
        });
        dbg.add_base_type(BaseType::i32());

        let info = dbg.emit_debug_info();
        let unit_length = u32::from_le_bytes(info[0..4].try_into().unwrap());
        assert_eq!(
            unit_length as usize,
            info.len() - 4,
            "unit_length should match actual debug_info size with lexical blocks"
        );
    }

    #[test]
    fn test_lexical_block_scoped_variable_in_string_table() {
        let mut dbg = DwarfDebugInfo::new("test.rs", "/tmp", SourceLanguage::Rust);
        dbg.add_function(FunctionDebugInfo {
            name: "baz".to_string(),
            linkage_name: None,
            low_pc: 0,
            size: 20,
            params: vec![],
            variables: vec![],
            scopes: vec![LexicalBlock {
                low_pc: 4,
                size: 12,
                variables: vec![VariableDebugInfo {
                    name: "scoped_var".to_string(),
                    type_index: 0,
                    location: VariableLocation::Register(0),
                }],
            }],
            line_entries: vec![],
            decl_file: 1,
            decl_line: 1,
        });
        dbg.add_base_type(BaseType::i32());

        let str_data = dbg.emit_debug_str();
        assert!(
            str_data.windows(b"scoped_var".len()).any(|w| w == b"scoped_var"),
            "string table should contain scoped variable name"
        );
    }

    #[test]
    fn test_abbrev_includes_lexical_block_entry() {
        let dbg = make_simple_debug_info();
        let abbrev = dbg.emit_abbrev();
        // ABBREV_LEXICAL_BLOCK=8, DW_TAG_LEXICAL_BLOCK=0x0b, DW_CHILDREN_YES=1
        assert!(
            abbrev.windows(3).any(|w| {
                w == [ABBREV_LEXICAL_BLOCK as u8, DW_TAG_LEXICAL_BLOCK as u8, DW_CHILDREN_YES]
            }),
            "abbreviation table should contain lexical block entry"
        );
    }

    // --- Linkage name tests ---

    #[test]
    fn test_linkage_name_in_string_table() {
        let mut dbg = DwarfDebugInfo::new("test.rs", "/tmp", SourceLanguage::Rust);
        dbg.add_function(FunctionDebugInfo {
            name: "add".to_string(),
            linkage_name: Some("_ZN4test3addEii".to_string()),
            low_pc: 0,
            size: 8,
            params: vec![],
            variables: vec![],
            scopes: vec![],
            line_entries: vec![],
            decl_file: 1,
            decl_line: 1,
        });
        dbg.add_base_type(BaseType::i32());

        let str_data = dbg.emit_debug_str();
        assert!(
            str_data
                .windows(b"_ZN4test3addEii".len())
                .any(|w| w == b"_ZN4test3addEii"),
            "string table should contain linkage name"
        );
    }

    // ---------------------------------------------------------------------
    // StructType tests (issue #326)
    // ---------------------------------------------------------------------

    fn make_point_struct_dbg() -> DwarfDebugInfo {
        // `struct Point { x: i32, y: i32 }` — two i32 members, 8 bytes.
        let mut dbg = DwarfDebugInfo::new(
            "point.rs",
            "./project",
            SourceLanguage::Rust,
        );
        dbg.add_base_type(BaseType::i32()); // type_index 0
        dbg.add_struct_type(StructType {
            name: "Point".to_string(),
            byte_size: 8,
            members: vec![
                StructMember { name: "x".to_string(), offset: 0, type_index: 0 },
                StructMember { name: "y".to_string(), offset: 4, type_index: 0 },
            ],
        });
        dbg
    }

    #[test]
    fn test_struct_type_adds_abbreviations() {
        let dbg = make_point_struct_dbg();
        let abbrev = dbg.emit_abbrev();
        // Abbreviation 9 (ULEB128 single byte 0x09) must appear followed by
        // the structure_type tag ULEB128 (0x13).
        assert!(
            abbrev.windows(2).any(|w| w == [0x09, 0x13]),
            "abbrev table should contain DW_TAG_structure_type abbrev (id 9)"
        );
        // Abbreviation 10 (ULEB128 single byte 0x0a) followed by DW_TAG_member
        // tag (0x0d).
        assert!(
            abbrev.windows(2).any(|w| w == [0x0a, 0x0d]),
            "abbrev table should contain DW_TAG_member abbrev (id 10)"
        );
    }

    #[test]
    fn test_struct_type_string_table_contains_names() {
        let dbg = make_point_struct_dbg();
        let str_tab = dbg.emit_debug_str();
        for needle in ["Point", "x", "y"] {
            assert!(
                str_tab.windows(needle.len()).any(|w| w == needle.as_bytes()),
                "string table should contain struct name/member: {needle}"
            );
        }
    }

    #[test]
    fn test_struct_type_debug_info_contains_structure_die() {
        let dbg = make_point_struct_dbg();
        let info = dbg.emit_debug_info();
        // DW_TAG_structure_type abbreviation id is 9. It must appear in
        // .debug_info. The byte_size (data4) is 8 and first member's offset
        // (data4) is 0. Both are encoded LE.
        let has_struct_abbrev = info.contains(&(ABBREV_STRUCTURE_TYPE as u8));
        assert!(has_struct_abbrev, "debug_info must reference struct abbrev");
        // byte_size(8) as u32 LE = [8,0,0,0]
        assert!(
            info.windows(4).any(|w| w == [8u8, 0, 0, 0]),
            "debug_info should contain byte_size=8 data4 literal"
        );
    }

    #[test]
    fn test_struct_type_debug_info_size_grows_with_member_count() {
        let mut one_member = DwarfDebugInfo::new(
            "a.rs",
            "/tmp",
            SourceLanguage::Rust,
        );
        one_member.add_base_type(BaseType::i32());
        one_member.add_struct_type(StructType {
            name: "A".to_string(),
            byte_size: 4,
            members: vec![StructMember {
                name: "f".to_string(),
                offset: 0,
                type_index: 0,
            }],
        });

        let mut three_members = DwarfDebugInfo::new(
            "a.rs",
            "/tmp",
            SourceLanguage::Rust,
        );
        three_members.add_base_type(BaseType::i32());
        three_members.add_struct_type(StructType {
            name: "A".to_string(),
            byte_size: 12,
            members: vec![
                StructMember { name: "f".to_string(), offset: 0, type_index: 0 },
                StructMember { name: "g".to_string(), offset: 4, type_index: 0 },
                StructMember { name: "h".to_string(), offset: 8, type_index: 0 },
            ],
        });

        let small = one_member.emit_debug_info().len();
        let big = three_members.emit_debug_info().len();
        assert!(
            big > small,
            "struct with 3 members should produce larger .debug_info \
             than struct with 1 member (small={small}, big={big})"
        );
    }

    // ---------------------------------------------------------------------
    // EnumType tests (issue #326)
    // ---------------------------------------------------------------------

    fn make_color_enum_dbg() -> DwarfDebugInfo {
        // `enum Color { Red = 0, Green = 1, Blue = 2 }` with `u32` storage.
        let mut dbg = DwarfDebugInfo::new(
            "color.rs",
            "./project",
            SourceLanguage::Rust,
        );
        dbg.add_base_type(BaseType::u32()); // underlying_type_index 0
        dbg.add_enum_type(EnumType {
            name: "Color".to_string(),
            byte_size: 4,
            underlying_type_index: 0,
            enumerators: vec![
                Enumerator { name: "Red".to_string(), value: 0 },
                Enumerator { name: "Green".to_string(), value: 1 },
                Enumerator { name: "Blue".to_string(), value: 2 },
            ],
        });
        dbg
    }

    #[test]
    fn test_enum_type_adds_abbreviations() {
        let dbg = make_color_enum_dbg();
        let abbrev = dbg.emit_abbrev();
        // Abbreviation 11 (ULEB128 single byte 0x0b) followed by
        // DW_TAG_enumeration_type (0x04).
        assert!(
            abbrev.windows(2).any(|w| w == [0x0b, 0x04]),
            "abbrev table should contain DW_TAG_enumeration_type abbrev (id 11)"
        );
        // Abbreviation 12 (ULEB128 single byte 0x0c) followed by
        // DW_TAG_enumerator (0x28).
        assert!(
            abbrev.windows(2).any(|w| w == [0x0c, 0x28]),
            "abbrev table should contain DW_TAG_enumerator abbrev (id 12)"
        );
    }

    #[test]
    fn test_enum_type_string_table_contains_names() {
        let dbg = make_color_enum_dbg();
        let str_tab = dbg.emit_debug_str();
        for needle in ["Color", "Red", "Green", "Blue"] {
            assert!(
                str_tab.windows(needle.len()).any(|w| w == needle.as_bytes()),
                "string table should contain enum/enumerator name: {needle}"
            );
        }
    }

    #[test]
    fn test_enum_type_debug_info_contains_enum_die() {
        let dbg = make_color_enum_dbg();
        let info = dbg.emit_debug_info();
        let has_enum_abbrev = info.contains(&(ABBREV_ENUMERATION_TYPE as u8));
        assert!(has_enum_abbrev, "debug_info must reference enum abbrev");
    }

    #[test]
    fn test_enum_type_debug_info_size_grows_with_enumerator_count() {
        let mut one_enumerator = DwarfDebugInfo::new(
            "color.rs",
            "/tmp",
            SourceLanguage::Rust,
        );
        one_enumerator.add_base_type(BaseType::u32());
        one_enumerator.add_enum_type(EnumType {
            name: "Color".to_string(),
            byte_size: 4,
            underlying_type_index: 0,
            enumerators: vec![
                Enumerator { name: "Red".to_string(), value: 0 },
            ],
        });

        let three_enumerators = make_color_enum_dbg();

        let small = one_enumerator.emit_debug_info().len();
        let big = three_enumerators.emit_debug_info().len();
        assert!(
            big > small,
            "enum with 3 enumerators should produce larger .debug_info \
             than enum with 1 enumerator (small={small}, big={big})"
        );
    }
}
