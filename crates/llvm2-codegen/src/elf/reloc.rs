// llvm2-codegen/elf/reloc.rs - ELF64 relocation types and encoding
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Reference: ARM IHI 0056B, "ELF for the Arm 64-bit Architecture"
// Reference: System V AMD64 ABI, "Relocation Types"
//
// Elf64_Rela layout (24 bytes, little-endian):
//   r_offset: u64  — location at which to apply the relocation
//   r_info:   u64  — symbol index (upper 32) | relocation type (lower 32)
//   r_addend: i64  — constant addend used to compute the value

//! ELF64 relocation types and encoding for AArch64 and x86-64.
//!
//! This module defines the relocation entry structure (`Elf64Rela`) and
//! relocation type enums for both target architectures. Each relocation entry
//! is 24 bytes and uses the SHT_RELA format (with explicit addend).

use super::constants::*;

/// AArch64 ELF relocation types.
///
/// These correspond to the R_AARCH64_* constants defined in the AArch64 ELF ABI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum AArch64RelocType {
    /// No relocation.
    None = R_AARCH64_NONE,
    /// S + A — 64-bit absolute data relocation.
    Abs64 = R_AARCH64_ABS64,
    /// S + A — 32-bit absolute data relocation.
    Abs32 = R_AARCH64_ABS32,
    /// Page(S + A) - Page(P) — ADRP instruction, 21-bit PC-relative page offset.
    AdrPrelPgHi21 = R_AARCH64_ADR_PREL_PG_HI21,
    /// (S + A) & 0xFFF — ADD instruction, 12-bit page offset (no check).
    AddAbsLo12Nc = R_AARCH64_ADD_ABS_LO12_NC,
    /// S + A - P — B instruction, 26-bit PC-relative branch.
    Jump26 = R_AARCH64_JUMP26,
    /// S + A - P — BL instruction, 26-bit PC-relative call.
    Call26 = R_AARCH64_CALL26,
    /// (S + A) & 0xFFF — LDR/STR 8-bit unsigned offset.
    Ldst8AbsLo12Nc = R_AARCH64_LDST8_ABS_LO12_NC,
    /// (S + A) & 0xFFF >> 1 — LDR/STR 16-bit unsigned offset.
    Ldst16AbsLo12Nc = R_AARCH64_LDST16_ABS_LO12_NC,
    /// (S + A) & 0xFFF >> 2 — LDR/STR 32-bit unsigned offset.
    Ldst32AbsLo12Nc = R_AARCH64_LDST32_ABS_LO12_NC,
    /// (S + A) & 0xFFF >> 3 — LDR/STR 64-bit unsigned offset.
    Ldst64AbsLo12Nc = R_AARCH64_LDST64_ABS_LO12_NC,
    /// (S + A) & 0xFFF >> 4 — LDR/STR 128-bit unsigned offset.
    Ldst128AbsLo12Nc = R_AARCH64_LDST128_ABS_LO12_NC,
    /// Page(G(S)) - Page(P) — ADRP to GOT entry page.
    AdrGotPage = R_AARCH64_ADR_GOT_PAGE,
    /// G(S) & 0xFFF — LD64 from GOT entry, 12-bit page offset.
    Ld64GotLo12Nc = R_AARCH64_LD64_GOT_LO12_NC,
    /// Page(G(S)) - Page(P) — ADRP to TLS IE GOT entry.
    TlsieAdrGottprelPage21 = R_AARCH64_TLSIE_ADR_GOTTPREL_PAGE21,
    /// G(S) & 0xFFF — LD64 from TLS IE GOT entry.
    TlsieLd64GottprelLo12Nc = R_AARCH64_TLSIE_LD64_GOTTPREL_LO12_NC,
}

impl AArch64RelocType {
    /// Return the numeric relocation type value.
    pub fn to_type_val(self) -> u32 {
        self as u32
    }

    /// Decode a numeric value to an AArch64 relocation type.
    pub fn from_type_val(val: u32) -> Option<Self> {
        match val {
            R_AARCH64_NONE => Some(Self::None),
            R_AARCH64_ABS64 => Some(Self::Abs64),
            R_AARCH64_ABS32 => Some(Self::Abs32),
            R_AARCH64_ADR_PREL_PG_HI21 => Some(Self::AdrPrelPgHi21),
            R_AARCH64_ADD_ABS_LO12_NC => Some(Self::AddAbsLo12Nc),
            R_AARCH64_JUMP26 => Some(Self::Jump26),
            R_AARCH64_CALL26 => Some(Self::Call26),
            R_AARCH64_LDST8_ABS_LO12_NC => Some(Self::Ldst8AbsLo12Nc),
            R_AARCH64_LDST16_ABS_LO12_NC => Some(Self::Ldst16AbsLo12Nc),
            R_AARCH64_LDST32_ABS_LO12_NC => Some(Self::Ldst32AbsLo12Nc),
            R_AARCH64_LDST64_ABS_LO12_NC => Some(Self::Ldst64AbsLo12Nc),
            R_AARCH64_LDST128_ABS_LO12_NC => Some(Self::Ldst128AbsLo12Nc),
            R_AARCH64_ADR_GOT_PAGE => Some(Self::AdrGotPage),
            R_AARCH64_LD64_GOT_LO12_NC => Some(Self::Ld64GotLo12Nc),
            R_AARCH64_TLSIE_ADR_GOTTPREL_PAGE21 => Some(Self::TlsieAdrGottprelPage21),
            R_AARCH64_TLSIE_LD64_GOTTPREL_LO12_NC => Some(Self::TlsieLd64GottprelLo12Nc),
            _ => None,
        }
    }
}

/// x86-64 ELF relocation types.
///
/// These correspond to the R_X86_64_* constants defined in the AMD64 ABI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum X86_64RelocType {
    /// No relocation.
    None = R_X86_64_NONE,
    /// S + A — 64-bit absolute address.
    Abs64 = R_X86_64_64,
    /// S + A - P — 32-bit PC-relative.
    Pc32 = R_X86_64_PC32,
    /// G + A — 32-bit GOT offset.
    Got32 = R_X86_64_GOT32,
    /// L + A - P — 32-bit PLT-relative.
    Plt32 = R_X86_64_PLT32,
    /// G + GOT + A - P — 32-bit GOT-relative PC offset.
    GotPcRel = R_X86_64_GOTPCREL,
    /// (zero-extended) S + A — 32-bit unsigned absolute.
    Abs32 = R_X86_64_32,
    /// (sign-extended) S + A — 32-bit signed absolute.
    Abs32S = R_X86_64_32S,
    /// S + A — 16-bit absolute.
    Abs16 = R_X86_64_16,
    /// S + A - P — 16-bit PC-relative.
    Pc16 = R_X86_64_PC16,
    /// S + A — 8-bit absolute.
    Abs8 = R_X86_64_8,
    /// S + A - P — 8-bit PC-relative.
    Pc8 = R_X86_64_PC8,
    /// Relaxable GOT-relative PC offset.
    GotPcRelX = R_X86_64_GOTPCRELX,
    /// Relaxable GOT-relative PC offset with REX prefix.
    RexGotPcRelX = R_X86_64_REX_GOTPCRELX,
}

impl X86_64RelocType {
    /// Return the numeric relocation type value.
    pub fn to_type_val(self) -> u32 {
        self as u32
    }

    /// Decode a numeric value to an x86-64 relocation type.
    pub fn from_type_val(val: u32) -> Option<Self> {
        match val {
            R_X86_64_NONE => Some(Self::None),
            R_X86_64_64 => Some(Self::Abs64),
            R_X86_64_PC32 => Some(Self::Pc32),
            R_X86_64_GOT32 => Some(Self::Got32),
            R_X86_64_PLT32 => Some(Self::Plt32),
            R_X86_64_GOTPCREL => Some(Self::GotPcRel),
            R_X86_64_32 => Some(Self::Abs32),
            R_X86_64_32S => Some(Self::Abs32S),
            R_X86_64_16 => Some(Self::Abs16),
            R_X86_64_PC16 => Some(Self::Pc16),
            R_X86_64_8 => Some(Self::Abs8),
            R_X86_64_PC8 => Some(Self::Pc8),
            R_X86_64_GOTPCRELX => Some(Self::GotPcRelX),
            R_X86_64_REX_GOTPCRELX => Some(Self::RexGotPcRelX),
            _ => None,
        }
    }
}

/// An ELF64 relocation entry with addend (Elf64_Rela).
///
/// Layout (24 bytes, little-endian):
/// ```text
/// Offset  Size  Field
/// 0        8    r_offset  — location within the section to apply the relocation
/// 8        8    r_info    — symbol table index (upper 32 bits) | type (lower 32 bits)
/// 16       8    r_addend  — constant addend for computing the relocation value
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Elf64Rela {
    /// Byte offset within the section where the relocation applies.
    pub r_offset: u64,
    /// Symbol table index of the referenced symbol.
    pub symbol_index: u32,
    /// Relocation type (architecture-specific).
    pub reloc_type: u32,
    /// Constant addend.
    pub r_addend: i64,
}

impl Elf64Rela {
    /// Create a new relocation entry.
    pub fn new(offset: u64, symbol_index: u32, reloc_type: u32, addend: i64) -> Self {
        Self {
            r_offset: offset,
            symbol_index,
            reloc_type,
            r_addend: addend,
        }
    }

    /// Create an AArch64 relocation entry.
    pub fn aarch64(offset: u64, symbol_index: u32, kind: AArch64RelocType, addend: i64) -> Self {
        Self::new(offset, symbol_index, kind.to_type_val(), addend)
    }

    /// Create an x86-64 relocation entry.
    pub fn x86_64(offset: u64, symbol_index: u32, kind: X86_64RelocType, addend: i64) -> Self {
        Self::new(offset, symbol_index, kind.to_type_val(), addend)
    }

    /// Compute the packed r_info field.
    pub fn r_info(&self) -> u64 {
        elf64_r_info(self.symbol_index, self.reloc_type)
    }

    /// Encode to 24-byte little-endian representation.
    pub fn encode(&self) -> [u8; ELF64_RELA_SIZE] {
        let mut buf = [0u8; ELF64_RELA_SIZE];
        buf[0..8].copy_from_slice(&self.r_offset.to_le_bytes());
        buf[8..16].copy_from_slice(&self.r_info().to_le_bytes());
        buf[16..24].copy_from_slice(&self.r_addend.to_le_bytes());
        buf
    }

    /// Decode from 24-byte little-endian representation.
    pub fn decode(bytes: &[u8; ELF64_RELA_SIZE]) -> Self {
        let r_offset = u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3],
            bytes[4], bytes[5], bytes[6], bytes[7],
        ]);
        let r_info = u64::from_le_bytes([
            bytes[8], bytes[9], bytes[10], bytes[11],
            bytes[12], bytes[13], bytes[14], bytes[15],
        ]);
        let r_addend = i64::from_le_bytes([
            bytes[16], bytes[17], bytes[18], bytes[19],
            bytes[20], bytes[21], bytes[22], bytes[23],
        ]);

        Self {
            r_offset,
            symbol_index: elf64_r_sym(r_info),
            reloc_type: elf64_r_type(r_info),
            r_addend,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rela_encode_decode_roundtrip() {
        let rela = Elf64Rela::aarch64(0x100, 5, AArch64RelocType::Call26, 0);
        let encoded = rela.encode();
        assert_eq!(encoded.len(), 24);
        let decoded = Elf64Rela::decode(&encoded);
        assert_eq!(rela, decoded);
    }

    #[test]
    fn test_rela_with_addend() {
        let rela = Elf64Rela::aarch64(0x10, 2, AArch64RelocType::AdrPrelPgHi21, 0x1000);
        let encoded = rela.encode();
        let decoded = Elf64Rela::decode(&encoded);
        assert_eq!(decoded.r_addend, 0x1000);
        assert_eq!(decoded.symbol_index, 2);
        assert_eq!(decoded.reloc_type, R_AARCH64_ADR_PREL_PG_HI21);
    }

    #[test]
    fn test_rela_negative_addend() {
        let rela = Elf64Rela::x86_64(0x20, 1, X86_64RelocType::Pc32, -4);
        let encoded = rela.encode();
        let decoded = Elf64Rela::decode(&encoded);
        assert_eq!(decoded.r_addend, -4);
    }

    #[test]
    fn test_r_info_encoding() {
        let rela = Elf64Rela::new(0, 10, R_AARCH64_CALL26, 0);
        let info = rela.r_info();
        assert_eq!(elf64_r_sym(info), 10);
        assert_eq!(elf64_r_type(info), R_AARCH64_CALL26);
    }

    #[test]
    fn test_aarch64_reloc_types() {
        let types = [
            (AArch64RelocType::None, R_AARCH64_NONE),
            (AArch64RelocType::Abs64, R_AARCH64_ABS64),
            (AArch64RelocType::AdrPrelPgHi21, R_AARCH64_ADR_PREL_PG_HI21),
            (AArch64RelocType::AddAbsLo12Nc, R_AARCH64_ADD_ABS_LO12_NC),
            (AArch64RelocType::Jump26, R_AARCH64_JUMP26),
            (AArch64RelocType::Call26, R_AARCH64_CALL26),
            (AArch64RelocType::Ldst64AbsLo12Nc, R_AARCH64_LDST64_ABS_LO12_NC),
            (AArch64RelocType::AdrGotPage, R_AARCH64_ADR_GOT_PAGE),
            (AArch64RelocType::Ld64GotLo12Nc, R_AARCH64_LD64_GOT_LO12_NC),
            (AArch64RelocType::TlsieAdrGottprelPage21, R_AARCH64_TLSIE_ADR_GOTTPREL_PAGE21),
            (AArch64RelocType::TlsieLd64GottprelLo12Nc, R_AARCH64_TLSIE_LD64_GOTTPREL_LO12_NC),
        ];

        for (kind, expected_val) in &types {
            assert_eq!(kind.to_type_val(), *expected_val);
            assert_eq!(AArch64RelocType::from_type_val(*expected_val), Some(*kind));
        }
    }

    #[test]
    fn test_x86_64_reloc_types() {
        let types = [
            (X86_64RelocType::None, R_X86_64_NONE),
            (X86_64RelocType::Abs64, R_X86_64_64),
            (X86_64RelocType::Pc32, R_X86_64_PC32),
            (X86_64RelocType::Got32, R_X86_64_GOT32),
            (X86_64RelocType::Plt32, R_X86_64_PLT32),
            (X86_64RelocType::GotPcRel, R_X86_64_GOTPCREL),
            (X86_64RelocType::Abs32, R_X86_64_32),
            (X86_64RelocType::Abs32S, R_X86_64_32S),
            (X86_64RelocType::GotPcRelX, R_X86_64_GOTPCRELX),
            (X86_64RelocType::RexGotPcRelX, R_X86_64_REX_GOTPCRELX),
        ];

        for (kind, expected_val) in &types {
            assert_eq!(kind.to_type_val(), *expected_val);
            assert_eq!(X86_64RelocType::from_type_val(*expected_val), Some(*kind));
        }
    }

    #[test]
    fn test_unknown_reloc_type() {
        assert_eq!(AArch64RelocType::from_type_val(9999), None);
        assert_eq!(X86_64RelocType::from_type_val(9999), None);
    }

    #[test]
    fn test_x86_64_rela_roundtrip() {
        let rela = Elf64Rela::x86_64(0x50, 3, X86_64RelocType::Plt32, -4);
        let encoded = rela.encode();
        let decoded = Elf64Rela::decode(&encoded);
        assert_eq!(decoded.r_offset, 0x50);
        assert_eq!(decoded.symbol_index, 3);
        assert_eq!(decoded.reloc_type, R_X86_64_PLT32);
        assert_eq!(decoded.r_addend, -4);
    }

    #[test]
    fn test_gotpcrel_roundtrip() {
        let rela = Elf64Rela::x86_64(0x30, 7, X86_64RelocType::GotPcRel, -4);
        let encoded = rela.encode();
        let decoded = Elf64Rela::decode(&encoded);
        assert_eq!(rela, decoded);
    }

    #[test]
    fn test_aarch64_got_roundtrip() {
        let rela = Elf64Rela::aarch64(0x00, 4, AArch64RelocType::AdrGotPage, 0);
        let encoded = rela.encode();
        let decoded = Elf64Rela::decode(&encoded);
        assert_eq!(rela, decoded);

        let rela = Elf64Rela::aarch64(0x04, 4, AArch64RelocType::Ld64GotLo12Nc, 0);
        let encoded = rela.encode();
        let decoded = Elf64Rela::decode(&encoded);
        assert_eq!(rela, decoded);
    }

    #[test]
    fn test_aarch64_tls_roundtrip() {
        let rela = Elf64Rela::aarch64(0x00, 6, AArch64RelocType::TlsieAdrGottprelPage21, 0);
        let encoded = rela.encode();
        let decoded = Elf64Rela::decode(&encoded);
        assert_eq!(rela, decoded);

        let rela = Elf64Rela::aarch64(0x04, 6, AArch64RelocType::TlsieLd64GottprelLo12Nc, 0);
        let encoded = rela.encode();
        let decoded = Elf64Rela::decode(&encoded);
        assert_eq!(rela, decoded);
    }
}
