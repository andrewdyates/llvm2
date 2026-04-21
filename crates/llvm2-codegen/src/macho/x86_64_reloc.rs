// llvm2-codegen/macho/x86_64_reloc.rs - x86-64 Mach-O relocation encoding
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Reference: LLVM X86MachObjectWriter.cpp, <mach-o/x86_64/reloc.h>
// Mach-O relocation_info struct layout (little-endian, 8 bytes):
//   r_word0: r_address (int32)
//   r_word1: r_symbolnum[0:23] | r_pcrel[24] | r_length[25:26] | r_extern[27] | r_type[28:31]

//! x86-64 Mach-O relocation encoding and emission.
//!
//! Implements the `relocation_info` struct encoding for the x86-64 Mach-O
//! relocation types defined in `<mach-o/x86_64/reloc.h>`. Each relocation
//! is 8 bytes: a 4-byte address followed by a 4-byte packed field containing
//! the symbol index, PC-relative flag, size, extern flag, and relocation type.

/// x86-64 relocation types as defined by the Mach-O ABI.
///
/// These correspond to the `X86_64_RELOC_*` constants in `<mach-o/x86_64/reloc.h>`.
/// Reference: Apple Mach-O Programming Topics, "x86-64 Relocations"
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum X86_64RelocKind {
    /// `X86_64_RELOC_UNSIGNED` (0) - Absolute address relocation.
    /// Used for 64-bit absolute addresses in data sections (.quad directives).
    /// Also used for 32-bit unsigned addresses when length=2.
    Unsigned = 0,

    /// `X86_64_RELOC_SIGNED` (1) - Signed 32-bit displacement.
    /// Used for RIP-relative data access: `mov rax, [rip + disp32]`.
    /// PC-relative, 4-byte (length=2).
    Signed = 1,

    /// `X86_64_RELOC_BRANCH` (2) - Call/jump displacement.
    /// Used for `CALL rel32` and `JMP rel32` instructions.
    /// PC-relative, 4-byte (length=2).
    Branch = 2,

    /// `X86_64_RELOC_GOT_LOAD` (3) - GOT load via RIP-relative MOV.
    /// Used for `mov rax, [rip + got_slot]` (GOT-indirect data access).
    /// PC-relative, 4-byte (length=2).
    GotLoad = 3,

    /// `X86_64_RELOC_GOT` (4) - Other GOT references.
    /// Used for non-load GOT references (e.g., LEA of a GOT slot).
    /// PC-relative, 4-byte (length=2).
    Got = 4,

    /// `X86_64_RELOC_SUBTRACTOR` (5) - Symbol difference.
    /// Must be followed by X86_64_RELOC_UNSIGNED. Used for `A - B` expressions
    /// in data sections.
    Subtractor = 5,

    /// `X86_64_RELOC_SIGNED_1` (6) - Signed 32-bit displacement with -1 addend.
    /// Used when the immediate value at the relocation site is -1 (e.g., for
    /// `cmpb $0x0, symbol(%rip)` where the displacement is adjusted by -1).
    Signed1 = 6,

    /// `X86_64_RELOC_SIGNED_2` (7) - Signed 32-bit displacement with -2 addend.
    /// Used when the immediate is 2 bytes, adjusting the displacement by -2.
    Signed2 = 7,

    /// `X86_64_RELOC_SIGNED_4` (8) - Signed 32-bit displacement with -4 addend.
    /// Used when the immediate is 4 bytes, adjusting the displacement by -4.
    Signed4 = 8,

    /// `X86_64_RELOC_TLV` (9) - Thread-local variable reference.
    /// Used for `mov rax, [rip + tlv_descriptor]`.
    /// PC-relative, 4-byte (length=2).
    Tlv = 9,
}

impl X86_64RelocKind {
    /// Returns whether this relocation type is PC-relative.
    ///
    /// On x86-64, most relocations other than UNSIGNED and SUBTRACTOR are
    /// PC-relative.
    pub fn is_pc_relative(self) -> bool {
        !matches!(self, X86_64RelocKind::Unsigned | X86_64RelocKind::Subtractor)
    }

    /// Returns log2 of the relocation size in bytes.
    ///
    /// Most x86-64 relocations operate on 4-byte (log2=2) displacement fields.
    /// Unsigned pointers can be 8-byte (log2=3) for 64-bit addresses.
    pub fn default_log2_size(self) -> u8 {
        match self {
            X86_64RelocKind::Unsigned => 3, // 8 bytes for 64-bit pointers (default)
            _ => 2, // 4 bytes for displacement fields
        }
    }
}

/// A Mach-O relocation entry for x86-64.
///
/// This corresponds to the `relocation_info` struct in `<mach-o/reloc.h>`,
/// specialized for x86-64 relocation types.
///
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
pub struct X86_64Relocation {
    /// Byte offset within the section where the relocation applies.
    pub offset: u32,

    /// Symbol table index (when `is_extern` is true) or section ordinal
    /// (1-based, when `is_extern` is false).
    pub symbol_index: u32,

    /// The x86-64 relocation type.
    pub kind: X86_64RelocKind,

    /// Whether this relocation is PC-relative.
    pub pc_relative: bool,

    /// Log2 of the relocation size: 0=1 byte, 1=2 bytes, 2=4 bytes, 3=8 bytes.
    pub length: u8,

    /// Whether the symbol_index refers to an external symbol table entry (true)
    /// or a section ordinal (false).
    pub is_extern: bool,
}

impl X86_64Relocation {
    /// Create a new relocation with default settings for the given kind.
    pub fn new(offset: u32, symbol_index: u32, kind: X86_64RelocKind) -> Self {
        Self {
            offset,
            symbol_index,
            kind,
            pc_relative: kind.is_pc_relative(),
            length: kind.default_log2_size(),
            is_extern: true,
        }
    }

    /// Create a Branch relocation for CALL/JMP rel32 instructions.
    pub fn branch(offset: u32, symbol_index: u32) -> Self {
        Self::new(offset, symbol_index, X86_64RelocKind::Branch)
    }

    /// Create a Signed relocation for RIP-relative data access.
    pub fn signed(offset: u32, symbol_index: u32) -> Self {
        Self::new(offset, symbol_index, X86_64RelocKind::Signed)
    }

    /// Create a Signed1 relocation (signed with -1 addend).
    pub fn signed_1(offset: u32, symbol_index: u32) -> Self {
        Self::new(offset, symbol_index, X86_64RelocKind::Signed1)
    }

    /// Create a Signed2 relocation (signed with -2 addend).
    pub fn signed_2(offset: u32, symbol_index: u32) -> Self {
        Self::new(offset, symbol_index, X86_64RelocKind::Signed2)
    }

    /// Create a Signed4 relocation (signed with -4 addend).
    pub fn signed_4(offset: u32, symbol_index: u32) -> Self {
        Self::new(offset, symbol_index, X86_64RelocKind::Signed4)
    }

    /// Create an Unsigned (absolute pointer) relocation (64-bit).
    pub fn unsigned_ptr(offset: u32, symbol_index: u32) -> Self {
        Self {
            offset,
            symbol_index,
            kind: X86_64RelocKind::Unsigned,
            pc_relative: false,
            length: 3, // 8 bytes
            is_extern: true,
        }
    }

    /// Create a 32-bit unsigned relocation.
    pub fn unsigned_32(offset: u32, symbol_index: u32) -> Self {
        Self {
            offset,
            symbol_index,
            kind: X86_64RelocKind::Unsigned,
            pc_relative: false,
            length: 2, // 4 bytes
            is_extern: true,
        }
    }

    /// Create a GOT load relocation for RIP-relative MOV from GOT.
    pub fn got_load(offset: u32, symbol_index: u32) -> Self {
        Self::new(offset, symbol_index, X86_64RelocKind::GotLoad)
    }

    /// Create a GOT relocation for non-load GOT references.
    pub fn got(offset: u32, symbol_index: u32) -> Self {
        Self::new(offset, symbol_index, X86_64RelocKind::Got)
    }

    /// Create a TLV relocation for thread-local variable references.
    pub fn tlv(offset: u32, symbol_index: u32) -> Self {
        Self::new(offset, symbol_index, X86_64RelocKind::Tlv)
    }

    /// Create a section-relative relocation (is_extern=false).
    ///
    /// The `section_ordinal` is the 1-based section number.
    pub fn section_relative(
        offset: u32,
        section_ordinal: u32,
        kind: X86_64RelocKind,
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

/// Encode an x86-64 relocation to the Mach-O `relocation_info` binary format
/// (8 bytes, little-endian).
///
/// Layout (little-endian):
/// - `r_word0` (bytes 0-3): `r_address` -- offset within section
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
pub fn encode_x86_64_relocation(reloc: &X86_64Relocation) -> [u8; 8] {
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

/// Error type for x86-64 relocation decoding failures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct X86_64RelocDecodeError {
    /// The unknown relocation type value.
    pub type_val: u8,
}

impl core::fmt::Display for X86_64RelocDecodeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "unknown x86-64 relocation type: {}", self.type_val)
    }
}

/// Decode an x86-64 Mach-O relocation from its 8-byte binary representation.
///
/// This is the inverse of [`encode_x86_64_relocation`].
///
/// Returns an error if the relocation type field contains an unrecognized value.
pub fn decode_x86_64_relocation(bytes: &[u8; 8]) -> Result<X86_64Relocation, X86_64RelocDecodeError> {
    let r_word0 = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    let r_word1 = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);

    let symbol_index = r_word1 & 0x00FF_FFFF;
    let pc_relative = (r_word1 >> 24) & 1 != 0;
    let length = ((r_word1 >> 25) & 3) as u8;
    let is_extern = (r_word1 >> 27) & 1 != 0;
    let type_val = ((r_word1 >> 28) & 0xF) as u8;

    let kind = match type_val {
        0 => X86_64RelocKind::Unsigned,
        1 => X86_64RelocKind::Signed,
        2 => X86_64RelocKind::Branch,
        3 => X86_64RelocKind::GotLoad,
        4 => X86_64RelocKind::Got,
        5 => X86_64RelocKind::Subtractor,
        6 => X86_64RelocKind::Signed1,
        7 => X86_64RelocKind::Signed2,
        8 => X86_64RelocKind::Signed4,
        9 => X86_64RelocKind::Tlv,
        _ => return Err(X86_64RelocDecodeError { type_val }),
    };

    Ok(X86_64Relocation {
        offset: r_word0,
        symbol_index,
        kind,
        pc_relative,
        length,
        is_extern,
    })
}

/// Create a subtractor relocation pair for symbol differences (A - B) on x86-64.
///
/// Mach-O represents `A - B` as two consecutive relocations:
/// 1. `X86_64_RELOC_SUBTRACTOR` referencing symbol B
/// 2. `X86_64_RELOC_UNSIGNED` referencing symbol A
///
/// Returns `(subtractor_reloc, unsigned_reloc)` where the subtractor must be emitted first.
pub fn create_x86_64_subtractor_pair(
    offset: u32,
    symbol_a_index: u32,
    symbol_b_index: u32,
    log2_size: u8,
) -> (X86_64Relocation, X86_64Relocation) {
    let subtractor = X86_64Relocation {
        offset,
        symbol_index: symbol_b_index,
        kind: X86_64RelocKind::Subtractor,
        pc_relative: false,
        length: log2_size,
        is_extern: true,
    };

    let unsigned = X86_64Relocation {
        offset,
        symbol_index: symbol_a_index,
        kind: X86_64RelocKind::Unsigned,
        pc_relative: false,
        length: log2_size,
        is_extern: true,
    };

    (subtractor, unsigned)
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Roundtrip tests ---

    #[test]
    fn test_encode_decode_roundtrip_branch() {
        let reloc = X86_64Relocation::branch(0x100, 42);
        let encoded = encode_x86_64_relocation(&reloc);
        let decoded = decode_x86_64_relocation(&encoded).unwrap();
        assert_eq!(reloc, decoded);
    }

    #[test]
    fn test_encode_decode_roundtrip_signed() {
        let reloc = X86_64Relocation::signed(0x200, 7);
        let encoded = encode_x86_64_relocation(&reloc);
        let decoded = decode_x86_64_relocation(&encoded).unwrap();
        assert_eq!(reloc, decoded);
    }

    #[test]
    fn test_encode_decode_roundtrip_unsigned() {
        let reloc = X86_64Relocation::unsigned_ptr(0x400, 99);
        let encoded = encode_x86_64_relocation(&reloc);
        let decoded = decode_x86_64_relocation(&encoded).unwrap();
        assert_eq!(reloc, decoded);
    }

    #[test]
    fn test_encode_decode_roundtrip_unsigned_32() {
        let reloc = X86_64Relocation::unsigned_32(0x300, 10);
        let encoded = encode_x86_64_relocation(&reloc);
        let decoded = decode_x86_64_relocation(&encoded).unwrap();
        assert_eq!(reloc, decoded);
    }

    #[test]
    fn test_encode_decode_roundtrip_got_load() {
        let reloc = X86_64Relocation::got_load(0x10, 3);
        let encoded = encode_x86_64_relocation(&reloc);
        let decoded = decode_x86_64_relocation(&encoded).unwrap();
        assert_eq!(reloc, decoded);
    }

    #[test]
    fn test_encode_decode_roundtrip_got() {
        let reloc = X86_64Relocation::got(0x14, 3);
        let encoded = encode_x86_64_relocation(&reloc);
        let decoded = decode_x86_64_relocation(&encoded).unwrap();
        assert_eq!(reloc, decoded);
    }

    #[test]
    fn test_encode_decode_roundtrip_signed_variants() {
        let reloc1 = X86_64Relocation::signed_1(0x10, 1);
        let encoded1 = encode_x86_64_relocation(&reloc1);
        assert_eq!(decode_x86_64_relocation(&encoded1).unwrap(), reloc1);

        let reloc2 = X86_64Relocation::signed_2(0x20, 2);
        let encoded2 = encode_x86_64_relocation(&reloc2);
        assert_eq!(decode_x86_64_relocation(&encoded2).unwrap(), reloc2);

        let reloc4 = X86_64Relocation::signed_4(0x30, 3);
        let encoded4 = encode_x86_64_relocation(&reloc4);
        assert_eq!(decode_x86_64_relocation(&encoded4).unwrap(), reloc4);
    }

    #[test]
    fn test_encode_decode_roundtrip_tlv() {
        let reloc = X86_64Relocation::tlv(0x20, 8);
        let encoded = encode_x86_64_relocation(&reloc);
        let decoded = decode_x86_64_relocation(&encoded).unwrap();
        assert_eq!(reloc, decoded);
        assert!(decoded.pc_relative);
        assert_eq!(decoded.kind, X86_64RelocKind::Tlv);
    }

    // --- Binary layout tests ---

    #[test]
    fn test_encode_branch_binary_layout() {
        // Branch: offset=0x10, symbol=5, pc_relative=true, length=2, extern=true, type=2
        let reloc = X86_64Relocation::branch(0x10, 5);
        let bytes = encode_x86_64_relocation(&reloc);

        // r_word0 = 0x10 (little-endian)
        assert_eq!(bytes[0..4], [0x10, 0x00, 0x00, 0x00]);

        let r_word1 = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        assert_eq!(r_word1 & 0x00FF_FFFF, 5, "symbol index");
        assert_eq!((r_word1 >> 24) & 1, 1, "pc_relative");
        assert_eq!((r_word1 >> 25) & 3, 2, "length");
        assert_eq!((r_word1 >> 27) & 1, 1, "is_extern");
        assert_eq!((r_word1 >> 28) & 0xF, 2, "type (Branch)");
    }

    #[test]
    fn test_encode_unsigned_binary_layout() {
        let reloc = X86_64Relocation::unsigned_ptr(0x20, 10);
        let bytes = encode_x86_64_relocation(&reloc);

        let r_word1 = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        assert_eq!(r_word1 & 0x00FF_FFFF, 10, "symbol index");
        assert_eq!((r_word1 >> 24) & 1, 0, "pc_relative (unsigned is not pcrel)");
        assert_eq!((r_word1 >> 25) & 3, 3, "length (3 = 8 bytes)");
        assert_eq!((r_word1 >> 27) & 1, 1, "is_extern");
        assert_eq!((r_word1 >> 28) & 0xF, 0, "type (Unsigned)");
    }

    #[test]
    fn test_encode_signed_binary_layout() {
        let reloc = X86_64Relocation::signed(0x04, 3);
        let bytes = encode_x86_64_relocation(&reloc);

        let r_word1 = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        assert_eq!(r_word1 & 0x00FF_FFFF, 3, "symbol index");
        assert_eq!((r_word1 >> 24) & 1, 1, "pc_relative");
        assert_eq!((r_word1 >> 25) & 3, 2, "length (2 = 4 bytes)");
        assert_eq!((r_word1 >> 27) & 1, 1, "is_extern");
        assert_eq!((r_word1 >> 28) & 0xF, 1, "type (Signed)");
    }

    // --- Error tests ---

    #[test]
    fn test_decode_unknown_reloc_type_returns_error() {
        let mut bytes = [0u8; 8];
        // r_word1 with type=15 (invalid for x86-64)
        let r_word1: u32 = 0xF000_0000;
        bytes[4..8].copy_from_slice(&r_word1.to_le_bytes());
        let result = decode_x86_64_relocation(&bytes);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().type_val, 15);
    }

    #[test]
    fn test_section_relative_relocation() {
        let reloc = X86_64Relocation::section_relative(0x50, 2, X86_64RelocKind::Unsigned);
        assert!(!reloc.is_extern);
        assert_eq!(reloc.symbol_index, 2);

        let bytes = encode_x86_64_relocation(&reloc);
        let r_word1 = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        assert_eq!((r_word1 >> 27) & 1, 0, "is_extern should be 0");
    }

    #[test]
    fn test_subtractor_pair() {
        let (sub_reloc, uns_reloc) = create_x86_64_subtractor_pair(0x20, 10, 5, 3);

        assert_eq!(sub_reloc.kind, X86_64RelocKind::Subtractor);
        assert_eq!(sub_reloc.symbol_index, 5); // B
        assert!(sub_reloc.is_extern);

        assert_eq!(uns_reloc.kind, X86_64RelocKind::Unsigned);
        assert_eq!(uns_reloc.symbol_index, 10); // A
        assert!(uns_reloc.is_extern);
    }

    // --- Kind property tests ---

    #[test]
    fn test_reloc_kind_properties() {
        assert!(X86_64RelocKind::Branch.is_pc_relative());
        assert!(X86_64RelocKind::Signed.is_pc_relative());
        assert!(X86_64RelocKind::Signed1.is_pc_relative());
        assert!(X86_64RelocKind::Signed2.is_pc_relative());
        assert!(X86_64RelocKind::Signed4.is_pc_relative());
        assert!(X86_64RelocKind::GotLoad.is_pc_relative());
        assert!(X86_64RelocKind::Got.is_pc_relative());
        assert!(X86_64RelocKind::Tlv.is_pc_relative());
        assert!(!X86_64RelocKind::Unsigned.is_pc_relative());
        assert!(!X86_64RelocKind::Subtractor.is_pc_relative());

        assert_eq!(X86_64RelocKind::Unsigned.default_log2_size(), 3);
        assert_eq!(X86_64RelocKind::Signed.default_log2_size(), 2);
        assert_eq!(X86_64RelocKind::Branch.default_log2_size(), 2);
        assert_eq!(X86_64RelocKind::GotLoad.default_log2_size(), 2);
        assert_eq!(X86_64RelocKind::Tlv.default_log2_size(), 2);
    }

    // --- Panic tests ---

    #[test]
    #[should_panic(expected = "symbol index")]
    fn test_encode_overflow_symbol_index() {
        let reloc = X86_64Relocation {
            offset: 0,
            symbol_index: 0x0100_0000,
            kind: X86_64RelocKind::Unsigned,
            pc_relative: false,
            length: 3,
            is_extern: true,
        };
        encode_x86_64_relocation(&reloc);
    }

    #[test]
    #[should_panic(expected = "length")]
    fn test_encode_overflow_length() {
        let reloc = X86_64Relocation {
            offset: 0,
            symbol_index: 0,
            kind: X86_64RelocKind::Unsigned,
            pc_relative: false,
            length: 4,
            is_extern: true,
        };
        encode_x86_64_relocation(&reloc);
    }
}
