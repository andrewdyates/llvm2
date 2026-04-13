// llvm2-codegen/x86_64/encode.rs - x86-64 instruction encoder (stub)
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Reference: ~/llvm-project-ref/llvm/lib/Target/X86/MCTargetDesc/X86MCCodeEmitter.cpp
// Reference: Intel 64 and IA-32 Architectures SDM, Volume 2

//! x86-64 instruction binary encoder.
//!
//! This is a stub implementation. Each method returns an error indicating
//! that x86-64 encoding is not yet implemented. The structure mirrors the
//! AArch64 encoder (`aarch64/encode.rs`) for consistency.

use llvm2_ir::x86_64_ops::X86Opcode;

/// Error type for x86-64 encoding failures.
#[derive(Debug, Clone)]
pub enum X86EncodeError {
    /// The opcode is not yet supported for encoding.
    UnsupportedOpcode(X86Opcode),
    /// The operand combination is invalid.
    InvalidOperands(String),
    /// x86-64 encoding is not yet implemented (scaffolding stub).
    NotImplemented(String),
}

impl core::fmt::Display for X86EncodeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::UnsupportedOpcode(op) => write!(f, "unsupported x86-64 opcode: {:?}", op),
            Self::InvalidOperands(msg) => write!(f, "invalid x86-64 operands: {}", msg),
            Self::NotImplemented(msg) => write!(f, "x86-64 not implemented: {}", msg),
        }
    }
}

// ---------------------------------------------------------------------------
// REX prefix builder
// ---------------------------------------------------------------------------

/// REX prefix flags for x86-64 encoding.
///
/// REX prefix byte: 0100 WRXB
/// - W: 1 = 64-bit operand size
/// - R: extension of ModR/M reg field
/// - X: extension of SIB index field
/// - B: extension of ModR/M r/m field, SIB base, or opcode reg
#[derive(Debug, Clone, Copy, Default)]
pub struct RexPrefix {
    /// REX.W: 64-bit operand size.
    pub w: bool,
    /// REX.R: ModR/M reg extension.
    pub r: bool,
    /// REX.X: SIB index extension.
    pub x: bool,
    /// REX.B: ModR/M r/m or opcode reg extension.
    pub b: bool,
}

impl RexPrefix {
    /// Returns true if a REX prefix is needed.
    pub fn is_needed(self) -> bool {
        self.w || self.r || self.x || self.b
    }

    /// Encode the REX prefix byte.
    pub fn encode(self) -> u8 {
        let mut byte: u8 = 0x40; // REX base
        if self.w { byte |= 0x08; }
        if self.r { byte |= 0x04; }
        if self.x { byte |= 0x02; }
        if self.b { byte |= 0x01; }
        byte
    }
}

// ---------------------------------------------------------------------------
// ModR/M byte builder
// ---------------------------------------------------------------------------

/// ModR/M byte encoding helper.
///
/// ModR/M byte layout: [mod:2][reg:3][rm:3]
#[derive(Debug, Clone, Copy)]
pub struct ModRM {
    /// Addressing mode (0b00 = [rm], 0b01 = [rm]+disp8, 0b10 = [rm]+disp32, 0b11 = register)
    pub mode: u8,
    /// Register operand or opcode extension (3 bits, lower 3 of 4-bit encoding).
    pub reg: u8,
    /// Register/memory operand (3 bits, lower 3 of 4-bit encoding).
    pub rm: u8,
}

impl ModRM {
    /// Create a register-register ModR/M (mod=11).
    pub fn reg_reg(reg: u8, rm: u8) -> Self {
        Self {
            mode: 0b11,
            reg: reg & 0x7,
            rm: rm & 0x7,
        }
    }

    /// Encode the ModR/M byte.
    pub fn encode(self) -> u8 {
        (self.mode << 6) | (self.reg << 3) | self.rm
    }
}

// ---------------------------------------------------------------------------
// X86Encoder — main encoder
// ---------------------------------------------------------------------------

/// x86-64 instruction encoder.
///
/// Encodes `X86Opcode` instructions into machine code bytes.
/// Currently a stub; all encoding methods return `NotImplemented`.
pub struct X86Encoder {
    /// Accumulated encoded bytes.
    pub bytes: Vec<u8>,
}

impl X86Encoder {
    /// Create a new empty encoder.
    pub fn new() -> Self {
        Self { bytes: Vec::new() }
    }

    /// Returns the encoded bytes.
    pub fn finish(self) -> Vec<u8> {
        self.bytes
    }

    /// Returns the current position (number of bytes emitted).
    pub fn position(&self) -> usize {
        self.bytes.len()
    }

    /// Emit a single byte.
    pub fn emit_byte(&mut self, byte: u8) {
        self.bytes.push(byte);
    }

    /// Emit a 32-bit little-endian value.
    pub fn emit_u32_le(&mut self, value: u32) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    /// Emit a 64-bit little-endian value.
    pub fn emit_u64_le(&mut self, value: u64) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    /// Emit a REX prefix if needed.
    pub fn emit_rex(&mut self, rex: RexPrefix) {
        if rex.is_needed() {
            self.emit_byte(rex.encode());
        }
    }

    /// Emit a ModR/M byte.
    pub fn emit_modrm(&mut self, modrm: ModRM) {
        self.emit_byte(modrm.encode());
    }

    /// Encode a single x86-64 instruction.
    ///
    /// TODO: Implement actual encoding for each opcode.
    /// This stub returns NotImplemented for all opcodes.
    pub fn encode_instruction(&mut self, opcode: X86Opcode) -> Result<(), X86EncodeError> {
        // Pseudo-instructions have no encoding
        if opcode.is_pseudo() {
            return Ok(());
        }

        Err(X86EncodeError::NotImplemented(
            format!("encoding for {:?} not yet implemented", opcode),
        ))
    }
}

impl Default for X86Encoder {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rex_prefix_not_needed() {
        let rex = RexPrefix::default();
        assert!(!rex.is_needed());
    }

    #[test]
    fn test_rex_prefix_w() {
        let rex = RexPrefix { w: true, ..Default::default() };
        assert!(rex.is_needed());
        assert_eq!(rex.encode(), 0x48);
    }

    #[test]
    fn test_rex_prefix_all() {
        let rex = RexPrefix { w: true, r: true, x: true, b: true };
        assert_eq!(rex.encode(), 0x4F);
    }

    #[test]
    fn test_rex_prefix_b_only() {
        let rex = RexPrefix { b: true, ..Default::default() };
        assert_eq!(rex.encode(), 0x41);
    }

    #[test]
    fn test_modrm_reg_reg() {
        // MOV RAX, RBX: mod=11, reg=RAX(0), rm=RBX(3)
        let modrm = ModRM::reg_reg(0, 3);
        assert_eq!(modrm.encode(), 0b11_000_011);
    }

    #[test]
    fn test_modrm_encode() {
        // mod=10, reg=5, rm=4 (SIB follows)
        let modrm = ModRM { mode: 0b10, reg: 5, rm: 4 };
        assert_eq!(modrm.encode(), 0b10_101_100);
    }

    #[test]
    fn test_encoder_new() {
        let enc = X86Encoder::new();
        assert_eq!(enc.position(), 0);
        assert!(enc.bytes.is_empty());
    }

    #[test]
    fn test_encoder_emit_bytes() {
        let mut enc = X86Encoder::new();
        enc.emit_byte(0x90); // NOP
        enc.emit_byte(0xC3); // RET
        assert_eq!(enc.position(), 2);
        assert_eq!(enc.finish(), vec![0x90, 0xC3]);
    }

    #[test]
    fn test_encoder_emit_u32() {
        let mut enc = X86Encoder::new();
        enc.emit_u32_le(0xDEADBEEF);
        assert_eq!(enc.finish(), vec![0xEF, 0xBE, 0xAD, 0xDE]);
    }

    #[test]
    fn test_encoder_emit_u64() {
        let mut enc = X86Encoder::new();
        enc.emit_u64_le(0x0102030405060708);
        assert_eq!(enc.finish(), vec![0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01]);
    }

    #[test]
    fn test_encoder_emit_rex() {
        let mut enc = X86Encoder::new();
        enc.emit_rex(RexPrefix { w: true, ..Default::default() });
        assert_eq!(enc.finish(), vec![0x48]);
    }

    #[test]
    fn test_encoder_emit_rex_not_needed() {
        let mut enc = X86Encoder::new();
        enc.emit_rex(RexPrefix::default());
        // No byte emitted when REX is not needed
        assert_eq!(enc.position(), 0);
    }

    #[test]
    fn test_encode_pseudo_succeeds() {
        let mut enc = X86Encoder::new();
        assert!(enc.encode_instruction(X86Opcode::Phi).is_ok());
        assert!(enc.encode_instruction(X86Opcode::Nop).is_ok());
        assert!(enc.encode_instruction(X86Opcode::StackAlloc).is_ok());
        assert_eq!(enc.position(), 0); // No bytes emitted for pseudos
    }

    #[test]
    fn test_encode_real_returns_not_implemented() {
        let mut enc = X86Encoder::new();
        let result = enc.encode_instruction(X86Opcode::AddRR);
        assert!(result.is_err());
        match result.unwrap_err() {
            X86EncodeError::NotImplemented(_) => {}
            other => panic!("Expected NotImplemented, got {:?}", other),
        }
    }

    #[test]
    fn test_encode_error_display() {
        let e1 = X86EncodeError::UnsupportedOpcode(X86Opcode::Ret);
        assert!(format!("{}", e1).contains("Ret"));

        let e2 = X86EncodeError::InvalidOperands("bad combo".into());
        assert!(format!("{}", e2).contains("bad combo"));

        let e3 = X86EncodeError::NotImplemented("stub".into());
        assert!(format!("{}", e3).contains("stub"));
    }
}
