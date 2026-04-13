// llvm2-codegen/macho/reloc.rs - ARM64 Mach-O relocation encoding
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Reference: LLVM AArch64MachObjectWriter.cpp, <mach-o/reloc.h>
// Mach-O relocation_info struct layout (little-endian, 8 bytes):
//   r_word0: r_address (int32)
//   r_word1: r_symbolnum[0:23] | r_pcrel[24] | r_length[25:26] | r_extern[27] | r_type[28:31]

//! ARM64 Mach-O relocation encoding and emission.
//!
//! Implements the `relocation_info` struct encoding for the AArch64 Mach-O
//! relocation types defined in `<mach-o/reloc.h>`. Each relocation is 8 bytes:
//! a 4-byte address followed by a 4-byte packed field containing the symbol
//! index, PC-relative flag, size, extern flag, and relocation type.

/// ARM64 relocation types as defined by the Mach-O ABI.
///
/// These correspond to the `ARM64_RELOC_*` constants in `<mach-o/reloc.h>`.
/// Each variant maps to a specific instruction encoding pattern used by the
/// AArch64 ISA for position-independent code and GOT access.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum AArch64RelocKind {
    /// `ARM64_RELOC_UNSIGNED` (0) - Absolute pointer relocation.
    /// Used for data pointers (64-bit absolute addresses in `.quad` directives).
    Unsigned = 0,

    /// `ARM64_RELOC_SUBTRACTOR` (1) - Symbol difference.
    /// Must be followed by an `ARM64_RELOC_UNSIGNED`. Used for `A - B` expressions
    /// in data sections.
    Subtractor = 1,

    /// `ARM64_RELOC_BRANCH26` (2) - 26-bit PC-relative branch offset.
    /// Used by B and BL instructions. The offset is a signed word offset
    /// (shifted left 2 to get byte offset), giving +/-128 MB range.
    Branch26 = 2,

    /// `ARM64_RELOC_PAGE21` (3) - 21-bit PC-relative page offset.
    /// Used by ADRP instruction. The value is a signed 21-bit page offset
    /// (shifted left 12 to get byte offset), giving +/-4 GB range.
    Page21 = 3,

    /// `ARM64_RELOC_PAGEOFF12` (4) - 12-bit page offset.
    /// Used by ADD and LDR instructions to compute the offset within a 4 KB page.
    /// Not PC-relative; combined with ADRP (Page21) for full addressing.
    Pageoff12 = 4,

    /// `ARM64_RELOC_GOT_LOAD_PAGE21` (5) - GOT entry page via ADRP.
    /// Like Page21 but references a GOT slot instead of the symbol directly.
    GotLoadPage21 = 5,

    /// `ARM64_RELOC_GOT_LOAD_PAGEOFF12` (6) - GOT entry page offset.
    /// Like Pageoff12 but references a GOT slot.
    GotLoadPageoff12 = 6,

    /// `ARM64_RELOC_POINTER_TO_GOT` (7) - 32-bit GOT-relative pointer.
    /// PC-relative pointer to a GOT entry, used in data sections.
    PointerToGot = 7,

    /// `ARM64_RELOC_TLVP_LOAD_PAGE21` (8) - TLV descriptor page.
    /// Like Page21 but for thread-local variable descriptors.
    TlvpLoadPage21 = 8,

    /// `ARM64_RELOC_TLVP_LOAD_PAGEOFF12` (9) - TLV descriptor page offset.
    /// Like Pageoff12 but for thread-local variable descriptors.
    TlvpLoadPageoff12 = 9,

    /// `ARM64_RELOC_ADDEND` (10) - Addend for complex relocations.
    /// Must be followed by `ARM64_RELOC_BRANCH26`, `ARM64_RELOC_PAGE21`,
    /// or `ARM64_RELOC_PAGEOFF12`. The symbolnum field encodes the addend value.
    Addend = 10,

    /// `ARM64_RELOC_AUTHENTICATED_POINTER` (11) - Authenticated pointer.
    /// Used for pointer authentication (PAC) on arm64e.
    AuthenticatedPointer = 11,
}

impl AArch64RelocKind {
    /// Returns whether this relocation type is PC-relative by default.
    ///
    /// Branch26 and Page21 are inherently PC-relative. Pageoff12 is not
    /// (it's an offset within a page, combined with an ADRP result).
    pub fn is_pc_relative(self) -> bool {
        matches!(
            self,
            AArch64RelocKind::Branch26
                | AArch64RelocKind::Page21
                | AArch64RelocKind::GotLoadPage21
                | AArch64RelocKind::TlvpLoadPage21
        )
    }

    /// Returns log2 of the relocation size in bytes.
    ///
    /// Most ARM64 relocations operate on 4-byte (log2=2) instruction fields.
    /// Unsigned pointers can be 8-byte (log2=3) for 64-bit addresses.
    pub fn default_log2_size(self) -> u8 {
        match self {
            AArch64RelocKind::Unsigned => 3, // 8 bytes for 64-bit pointers
            _ => 2,                          // 4 bytes for instruction-embedded fields
        }
    }
}

/// A Mach-O relocation entry for ARM64.
///
/// This corresponds to the `relocation_info` struct in `<mach-o/reloc.h>`:
/// ```text
/// struct relocation_info {
///     int32_t   r_address;     // offset in section
///     uint32_t  r_symbolnum:24, // symbol table index (or section ordinal if !extern)
///               r_pcrel:1,      // PC-relative flag
///               r_length:2,     // log2 of size (0=1B, 1=2B, 2=4B, 3=8B)
///               r_extern:1,     // external symbol flag
///               r_type:4;       // relocation type
/// };
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Relocation {
    /// Byte offset within the section where the relocation applies.
    pub offset: u32,

    /// Symbol table index (when `is_extern` is true) or section ordinal
    /// (1-based, when `is_extern` is false).
    pub symbol_index: u32,

    /// The ARM64 relocation type.
    pub kind: AArch64RelocKind,

    /// Whether this relocation is PC-relative.
    pub pc_relative: bool,

    /// Log2 of the relocation size: 0=1 byte, 1=2 bytes, 2=4 bytes, 3=8 bytes.
    pub length: u8,

    /// Whether the symbol_index refers to an external symbol table entry (true)
    /// or a section ordinal (false).
    pub is_extern: bool,
}

impl Relocation {
    /// Create a new relocation with default settings for the given kind.
    ///
    /// Sets `pc_relative` and `length` based on the relocation kind's defaults.
    /// The caller can override these after construction if needed.
    pub fn new(offset: u32, symbol_index: u32, kind: AArch64RelocKind) -> Self {
        Self {
            offset,
            symbol_index,
            kind,
            pc_relative: kind.is_pc_relative(),
            length: kind.default_log2_size(),
            is_extern: true,
        }
    }

    /// Create a Branch26 relocation for B/BL instructions.
    pub fn branch26(offset: u32, symbol_index: u32) -> Self {
        Self::new(offset, symbol_index, AArch64RelocKind::Branch26)
    }

    /// Create a Page21 relocation for ADRP instructions.
    pub fn page21(offset: u32, symbol_index: u32) -> Self {
        Self::new(offset, symbol_index, AArch64RelocKind::Page21)
    }

    /// Create a Pageoff12 relocation for ADD/LDR page offset instructions.
    pub fn pageoff12(offset: u32, symbol_index: u32) -> Self {
        Self::new(offset, symbol_index, AArch64RelocKind::Pageoff12)
    }

    /// Create a GOT load page relocation for ADRP to GOT slot.
    pub fn got_load_page21(offset: u32, symbol_index: u32) -> Self {
        Self::new(offset, symbol_index, AArch64RelocKind::GotLoadPage21)
    }

    /// Create a GOT load page offset relocation.
    pub fn got_load_pageoff12(offset: u32, symbol_index: u32) -> Self {
        Self::new(offset, symbol_index, AArch64RelocKind::GotLoadPageoff12)
    }

    /// Create a TLV page relocation for ADRP to TLV descriptor.
    pub fn tlvp_load_page21(offset: u32, symbol_index: u32) -> Self {
        Self::new(offset, symbol_index, AArch64RelocKind::TlvpLoadPage21)
    }

    /// Create a TLV page offset relocation for LDR from TLV descriptor.
    pub fn tlvp_load_pageoff12(offset: u32, symbol_index: u32) -> Self {
        Self::new(offset, symbol_index, AArch64RelocKind::TlvpLoadPageoff12)
    }

    /// Create an Unsigned (absolute pointer) relocation.
    pub fn unsigned_ptr(offset: u32, symbol_index: u32) -> Self {
        Self {
            offset,
            symbol_index,
            kind: AArch64RelocKind::Unsigned,
            pc_relative: false,
            length: 3, // 8 bytes
            is_extern: true,
        }
    }

    /// Create a section-relative relocation (is_extern=false).
    ///
    /// The `section_ordinal` is the 1-based section number.
    pub fn section_relative(
        offset: u32,
        section_ordinal: u32,
        kind: AArch64RelocKind,
    ) -> Self {
        Self {
            offset,
            symbol_index: section_ordinal,
            kind,
            pc_relative: kind.is_pc_relative(),
            length: kind.default_log2_size(),
            is_extern: false,
        }
    }
}

/// Encode a relocation to the Mach-O `relocation_info` binary format (8 bytes, little-endian).
///
/// Layout (little-endian):
/// - `r_word0` (bytes 0-3): `r_address` — offset within section
/// - `r_word1` (bytes 4-7): packed bitfield:
///   - bits  0-23: `r_symbolnum` (symbol index or section ordinal)
///   - bit     24: `r_pcrel` (1 = PC-relative)
///   - bits 25-26: `r_length` (log2 of size)
///   - bit     27: `r_extern` (1 = external symbol)
///   - bits 28-31: `r_type` (relocation type)
///
/// # Panics
///
/// Panics if `symbol_index` exceeds 24 bits (> 0xFF_FFFF) or `length` exceeds
/// 2 bits (> 3).
pub fn encode_relocation(reloc: &Relocation) -> [u8; 8] {
    assert!(
        reloc.symbol_index <= 0x00FF_FFFF,
        "symbol index {} exceeds 24-bit field",
        reloc.symbol_index
    );
    assert!(
        reloc.length <= 3,
        "length {} exceeds 2-bit field (max log2 size is 3 for 8 bytes)",
        reloc.length
    );

    let r_word0: u32 = reloc.offset;
    let r_word1: u32 = (reloc.symbol_index & 0x00FF_FFFF)
        | ((reloc.pc_relative as u32) << 24)
        | ((reloc.length as u32) << 25)
        | ((reloc.is_extern as u32) << 27)
        | ((reloc.kind as u32) << 28);

    let mut buf = [0u8; 8];
    buf[0..4].copy_from_slice(&r_word0.to_le_bytes());
    buf[4..8].copy_from_slice(&r_word1.to_le_bytes());
    buf
}

/// Error type for relocation decoding failures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelocDecodeError {
    /// The unknown relocation type value.
    pub type_val: u8,
}

impl core::fmt::Display for RelocDecodeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "unknown ARM64 relocation type: {}", self.type_val)
    }
}

/// Decode a Mach-O relocation from its 8-byte binary representation (little-endian).
///
/// This is the inverse of [`encode_relocation`].
///
/// Returns an error if the relocation type field contains an unrecognized value.
pub fn decode_relocation(bytes: &[u8; 8]) -> Result<Relocation, RelocDecodeError> {
    let r_word0 = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    let r_word1 = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);

    let symbol_index = r_word1 & 0x00FF_FFFF;
    let pc_relative = (r_word1 >> 24) & 1 != 0;
    let length = ((r_word1 >> 25) & 3) as u8;
    let is_extern = (r_word1 >> 27) & 1 != 0;
    let type_val = ((r_word1 >> 28) & 0xF) as u8;

    let kind = match type_val {
        0 => AArch64RelocKind::Unsigned,
        1 => AArch64RelocKind::Subtractor,
        2 => AArch64RelocKind::Branch26,
        3 => AArch64RelocKind::Page21,
        4 => AArch64RelocKind::Pageoff12,
        5 => AArch64RelocKind::GotLoadPage21,
        6 => AArch64RelocKind::GotLoadPageoff12,
        7 => AArch64RelocKind::PointerToGot,
        8 => AArch64RelocKind::TlvpLoadPage21,
        9 => AArch64RelocKind::TlvpLoadPageoff12,
        10 => AArch64RelocKind::Addend,
        11 => AArch64RelocKind::AuthenticatedPointer,
        _ => return Err(RelocDecodeError { type_val }),
    };

    Ok(Relocation {
        offset: r_word0,
        symbol_index,
        kind,
        pc_relative,
        length,
        is_extern,
    })
}

/// Create an addend relocation pair.
///
/// ARM64 Mach-O encodes addends for `Branch26`, `Page21`, and `Pageoff12`
/// as a separate `ARM64_RELOC_ADDEND` relocation placed immediately before
/// the actual relocation. The addend value is stored in the `r_symbolnum` field.
///
/// Returns `(addend_reloc, main_reloc)` where the addend must be emitted first.
///
/// # Panics
///
/// Panics if the addend does not fit in a signed 24-bit value.
pub fn create_addend_pair(
    offset: u32,
    symbol_index: u32,
    kind: AArch64RelocKind,
    addend: i32,
) -> (Relocation, Relocation) {
    assert!(
        matches!(
            kind,
            AArch64RelocKind::Branch26
                | AArch64RelocKind::Page21
                | AArch64RelocKind::Pageoff12
        ),
        "addend relocations only supported for Branch26, Page21, Pageoff12"
    );
    // Addend must fit in signed 24-bit (per LLVM: isInt<24>)
    assert!(
        addend >= -(1 << 23) && addend < (1 << 23),
        "addend {} does not fit in signed 24-bit field",
        addend
    );

    let addend_reloc = Relocation {
        offset,
        // The addend value is stored in r_symbolnum (treated as signed 24-bit)
        symbol_index: (addend as u32) & 0x00FF_FFFF,
        kind: AArch64RelocKind::Addend,
        pc_relative: false,
        length: 2, // log2(4) — matches LLVM
        is_extern: false,
    };

    let main_reloc = Relocation::new(offset, symbol_index, kind);

    (addend_reloc, main_reloc)
}

/// Create a subtractor relocation pair for symbol differences (A - B).
///
/// Mach-O represents `A - B` as two consecutive relocations:
/// 1. `ARM64_RELOC_SUBTRACTOR` referencing symbol B
/// 2. `ARM64_RELOC_UNSIGNED` referencing symbol A
///
/// Returns `(subtractor_reloc, unsigned_reloc)` where the subtractor must be emitted first.
pub fn create_subtractor_pair(
    offset: u32,
    symbol_a_index: u32,
    symbol_b_index: u32,
    log2_size: u8,
) -> (Relocation, Relocation) {
    let subtractor = Relocation {
        offset,
        symbol_index: symbol_b_index,
        kind: AArch64RelocKind::Subtractor,
        pc_relative: false,
        length: log2_size,
        is_extern: true,
    };

    let unsigned = Relocation {
        offset,
        symbol_index: symbol_a_index,
        kind: AArch64RelocKind::Unsigned,
        pc_relative: false,
        length: log2_size,
        is_extern: true,
    };

    (subtractor, unsigned)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode_roundtrip_branch26() {
        let reloc = Relocation::branch26(0x100, 42);
        let encoded = encode_relocation(&reloc);
        let decoded = decode_relocation(&encoded).unwrap();
        assert_eq!(reloc, decoded);
    }

    #[test]
    fn test_encode_decode_roundtrip_page21() {
        let reloc = Relocation::page21(0x200, 7);
        let encoded = encode_relocation(&reloc);
        let decoded = decode_relocation(&encoded).unwrap();
        assert_eq!(reloc, decoded);
    }

    #[test]
    fn test_encode_decode_roundtrip_pageoff12() {
        let reloc = Relocation::pageoff12(0x300, 15);
        let encoded = encode_relocation(&reloc);
        let decoded = decode_relocation(&encoded).unwrap();
        assert_eq!(reloc, decoded);
    }

    #[test]
    fn test_encode_decode_roundtrip_unsigned() {
        let reloc = Relocation::unsigned_ptr(0x400, 99);
        let encoded = encode_relocation(&reloc);
        let decoded = decode_relocation(&encoded).unwrap();
        assert_eq!(reloc, decoded);
    }

    #[test]
    fn test_encode_decode_roundtrip_got() {
        let reloc = Relocation::got_load_page21(0x10, 3);
        let encoded = encode_relocation(&reloc);
        let decoded = decode_relocation(&encoded).unwrap();
        assert_eq!(reloc, decoded);

        let reloc = Relocation::got_load_pageoff12(0x14, 3);
        let encoded = encode_relocation(&reloc);
        let decoded = decode_relocation(&encoded).unwrap();
        assert_eq!(reloc, decoded);
    }

    #[test]
    fn test_decode_unknown_reloc_type_returns_error() {
        // Craft bytes with type_val = 15 (invalid)
        let mut bytes = [0u8; 8];
        // r_word1: type=15 in bits 28-31
        let r_word1: u32 = 0xF000_0000;
        bytes[4..8].copy_from_slice(&r_word1.to_le_bytes());
        let result = decode_relocation(&bytes);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().type_val, 15);
    }

    #[test]
    fn test_encode_branch26_binary_layout() {
        // Branch26: offset=0x10, symbol=5, pc_relative=true, length=2, extern=true, type=2
        let reloc = Relocation::branch26(0x10, 5);
        let bytes = encode_relocation(&reloc);

        // r_word0 = 0x10 (little-endian)
        assert_eq!(bytes[0..4], [0x10, 0x00, 0x00, 0x00]);

        // r_word1: symbolnum=5, pcrel=1, length=2(0b10), extern=1, type=2(0b0010)
        // bits: [5:24] | [1:1] | [2:2] | [1:1] | [2:4]
        // = 0x05 | (1 << 24) | (2 << 25) | (1 << 27) | (2 << 28)
        // = 0x05 | 0x0100_0000 | 0x0400_0000 | 0x0800_0000 | 0x2000_0000
        // = 0x2D00_0005
        let r_word1 = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        assert_eq!(r_word1 & 0x00FF_FFFF, 5, "symbol index");
        assert_eq!((r_word1 >> 24) & 1, 1, "pc_relative");
        assert_eq!((r_word1 >> 25) & 3, 2, "length");
        assert_eq!((r_word1 >> 27) & 1, 1, "is_extern");
        assert_eq!((r_word1 >> 28) & 0xF, 2, "type (Branch26)");
    }

    #[test]
    fn test_encode_unsigned_binary_layout() {
        let reloc = Relocation::unsigned_ptr(0x20, 10);
        let bytes = encode_relocation(&reloc);

        let r_word1 = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        assert_eq!(r_word1 & 0x00FF_FFFF, 10, "symbol index");
        assert_eq!((r_word1 >> 24) & 1, 0, "pc_relative (unsigned is not pcrel)");
        assert_eq!((r_word1 >> 25) & 3, 3, "length (3 = 8 bytes)");
        assert_eq!((r_word1 >> 27) & 1, 1, "is_extern");
        assert_eq!((r_word1 >> 28) & 0xF, 0, "type (Unsigned)");
    }

    #[test]
    fn test_section_relative_relocation() {
        let reloc = Relocation::section_relative(0x50, 2, AArch64RelocKind::Unsigned);
        assert!(!reloc.is_extern);
        assert_eq!(reloc.symbol_index, 2);

        let bytes = encode_relocation(&reloc);
        let r_word1 = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        assert_eq!((r_word1 >> 27) & 1, 0, "is_extern should be 0");
    }

    #[test]
    fn test_addend_pair() {
        let (addend_reloc, main_reloc) =
            create_addend_pair(0x10, 5, AArch64RelocKind::Branch26, 0x100);

        assert_eq!(addend_reloc.kind, AArch64RelocKind::Addend);
        assert_eq!(addend_reloc.symbol_index, 0x100);
        assert!(!addend_reloc.pc_relative);

        assert_eq!(main_reloc.kind, AArch64RelocKind::Branch26);
        assert_eq!(main_reloc.symbol_index, 5);
        assert!(main_reloc.pc_relative);
    }

    #[test]
    fn test_addend_pair_negative() {
        let (addend_reloc, _main_reloc) =
            create_addend_pair(0x10, 5, AArch64RelocKind::Page21, -4);

        // -4 as unsigned 24-bit: 0xFFFFFC & 0xFFFFFF = 0xFFFFFC
        assert_eq!(addend_reloc.symbol_index, (-4i32 as u32) & 0x00FF_FFFF);
    }

    #[test]
    fn test_subtractor_pair() {
        let (sub_reloc, uns_reloc) = create_subtractor_pair(0x20, 10, 5, 3);

        assert_eq!(sub_reloc.kind, AArch64RelocKind::Subtractor);
        assert_eq!(sub_reloc.symbol_index, 5); // B
        assert!(sub_reloc.is_extern);

        assert_eq!(uns_reloc.kind, AArch64RelocKind::Unsigned);
        assert_eq!(uns_reloc.symbol_index, 10); // A
        assert!(uns_reloc.is_extern);
    }

    #[test]
    fn test_reloc_kind_properties() {
        assert!(AArch64RelocKind::Branch26.is_pc_relative());
        assert!(AArch64RelocKind::Page21.is_pc_relative());
        assert!(!AArch64RelocKind::Pageoff12.is_pc_relative());
        assert!(!AArch64RelocKind::Unsigned.is_pc_relative());
        assert!(AArch64RelocKind::GotLoadPage21.is_pc_relative());
        assert!(!AArch64RelocKind::GotLoadPageoff12.is_pc_relative());
        assert!(AArch64RelocKind::TlvpLoadPage21.is_pc_relative());
        assert!(!AArch64RelocKind::TlvpLoadPageoff12.is_pc_relative());

        assert_eq!(AArch64RelocKind::Unsigned.default_log2_size(), 3);
        assert_eq!(AArch64RelocKind::Branch26.default_log2_size(), 2);
        assert_eq!(AArch64RelocKind::Page21.default_log2_size(), 2);
        assert_eq!(AArch64RelocKind::Pageoff12.default_log2_size(), 2);
        assert_eq!(AArch64RelocKind::TlvpLoadPage21.default_log2_size(), 2);
        assert_eq!(AArch64RelocKind::TlvpLoadPageoff12.default_log2_size(), 2);
    }

    #[test]
    fn test_encode_decode_roundtrip_tlvp() {
        let reloc = Relocation::tlvp_load_page21(0x20, 8);
        let encoded = encode_relocation(&reloc);
        let decoded = decode_relocation(&encoded).unwrap();
        assert_eq!(reloc, decoded);
        assert!(decoded.pc_relative);
        assert_eq!(decoded.kind, AArch64RelocKind::TlvpLoadPage21);

        let reloc = Relocation::tlvp_load_pageoff12(0x24, 8);
        let encoded = encode_relocation(&reloc);
        let decoded = decode_relocation(&encoded).unwrap();
        assert_eq!(reloc, decoded);
        assert!(!decoded.pc_relative);
        assert_eq!(decoded.kind, AArch64RelocKind::TlvpLoadPageoff12);
    }

    #[test]
    fn test_tlvp_reloc_binary_layout() {
        // TlvpLoadPage21: offset=0x20, symbol=8, pcrel=1, length=2, extern=1, type=8
        let reloc = Relocation::tlvp_load_page21(0x20, 8);
        let bytes = encode_relocation(&reloc);

        let r_word1 = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        assert_eq!(r_word1 & 0x00FF_FFFF, 8, "symbol index");
        assert_eq!((r_word1 >> 24) & 1, 1, "pc_relative");
        assert_eq!((r_word1 >> 25) & 3, 2, "length");
        assert_eq!((r_word1 >> 27) & 1, 1, "is_extern");
        assert_eq!((r_word1 >> 28) & 0xF, 8, "type (TlvpLoadPage21)");

        // TlvpLoadPageoff12: offset=0x24, symbol=8, pcrel=0, length=2, extern=1, type=9
        let reloc = Relocation::tlvp_load_pageoff12(0x24, 8);
        let bytes = encode_relocation(&reloc);

        let r_word1 = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        assert_eq!(r_word1 & 0x00FF_FFFF, 8, "symbol index");
        assert_eq!((r_word1 >> 24) & 1, 0, "pc_relative (pageoff is not pcrel)");
        assert_eq!((r_word1 >> 25) & 3, 2, "length");
        assert_eq!((r_word1 >> 27) & 1, 1, "is_extern");
        assert_eq!((r_word1 >> 28) & 0xF, 9, "type (TlvpLoadPageoff12)");
    }

    #[test]
    #[should_panic(expected = "symbol index")]
    fn test_encode_overflow_symbol_index() {
        let reloc = Relocation {
            offset: 0,
            symbol_index: 0x0100_0000, // 25 bits, exceeds 24-bit field
            kind: AArch64RelocKind::Unsigned,
            pc_relative: false,
            length: 3,
            is_extern: true,
        };
        encode_relocation(&reloc);
    }

    #[test]
    #[should_panic(expected = "length")]
    fn test_encode_overflow_length() {
        let reloc = Relocation {
            offset: 0,
            symbol_index: 0,
            kind: AArch64RelocKind::Unsigned,
            pc_relative: false,
            length: 4, // max is 3
            is_extern: true,
        };
        encode_relocation(&reloc);
    }
}
