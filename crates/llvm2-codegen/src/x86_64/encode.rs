// llvm2-codegen/x86_64/encode.rs - x86-64 instruction binary encoder
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0
//
// Reference: ~/llvm-project-ref/llvm/lib/Target/X86/MCTargetDesc/X86MCCodeEmitter.cpp
// Reference: Intel 64 and IA-32 Architectures SDM, Volume 2

//! x86-64 instruction binary encoder.
//!
//! Encodes `X86Opcode` instructions into variable-length machine code bytes.
//! Each encoding method produces the correct prefix, opcode, ModR/M, SIB,
//! displacement, and immediate bytes per the Intel SDM Vol 2.
//!
//! # Encoding format (general structure)
//!
//! ```text
//! [Legacy prefix] [REX prefix] [Opcode 1-3 bytes] [ModR/M] [SIB] [Disp] [Imm]
//! ```
//!
//! # REX prefix byte: `0100 WRXB`
//!
//! - W: 1 = 64-bit operand size
//! - R: extension of ModR/M reg field (bit 3)
//! - X: extension of SIB index field (bit 3)
//! - B: extension of ModR/M r/m field, SIB base, or opcode reg (bit 3)

use llvm2_ir::x86_64_ops::{X86CondCode, X86Opcode};
use llvm2_ir::x86_64_regs::X86PReg;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Error type for x86-64 encoding failures.
#[derive(Debug, Clone)]
pub enum X86EncodeError {
    /// The opcode is not yet supported for encoding.
    UnsupportedOpcode(X86Opcode),
    /// The operand combination is invalid.
    InvalidOperands(String),
    /// x86-64 encoding is not yet implemented for this specific form.
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
/// REX prefix byte: `0100 WRXB`
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
        if self.w {
            byte |= 0x08;
        }
        if self.r {
            byte |= 0x04;
        }
        if self.x {
            byte |= 0x02;
        }
        if self.b {
            byte |= 0x01;
        }
        byte
    }
}

// ---------------------------------------------------------------------------
// ModR/M byte builder
// ---------------------------------------------------------------------------

/// ModR/M byte encoding helper.
///
/// ModR/M byte layout: `[mod:2][reg:3][rm:3]`
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

    /// Create ModR/M for opcode extension with register operand (mod=11).
    pub fn ext_reg(ext: u8, rm: u8) -> Self {
        Self {
            mode: 0b11,
            reg: ext & 0x7,
            rm: rm & 0x7,
        }
    }

    /// Create ModR/M for [base] addressing (mod=00), no displacement.
    pub fn indirect(reg: u8, base: u8) -> Self {
        Self {
            mode: 0b00,
            reg: reg & 0x7,
            rm: base & 0x7,
        }
    }

    /// Create ModR/M for [base+disp8] addressing (mod=01).
    pub fn indirect_disp8(reg: u8, base: u8) -> Self {
        Self {
            mode: 0b01,
            reg: reg & 0x7,
            rm: base & 0x7,
        }
    }

    /// Create ModR/M for [base+disp32] addressing (mod=10).
    pub fn indirect_disp32(reg: u8, base: u8) -> Self {
        Self {
            mode: 0b10,
            reg: reg & 0x7,
            rm: base & 0x7,
        }
    }

    /// Encode the ModR/M byte.
    pub fn encode(self) -> u8 {
        (self.mode << 6) | (self.reg << 3) | self.rm
    }
}

// ---------------------------------------------------------------------------
// SIB byte builder
// ---------------------------------------------------------------------------

/// SIB (Scale-Index-Base) byte encoding helper.
///
/// SIB byte layout: `[scale:2][index:3][base:3]`
///
/// Used when ModR/M rm=100 (RSP encoding) to specify complex addressing modes:
/// `[base + index * scale + displacement]`
#[derive(Debug, Clone, Copy)]
pub struct Sib {
    /// Scale factor: 0=1, 1=2, 2=4, 3=8.
    pub scale: u8,
    /// Index register (3 bits, lower 3 of 4-bit encoding). 0b100 = no index.
    pub index: u8,
    /// Base register (3 bits, lower 3 of 4-bit encoding).
    pub base: u8,
}

impl Sib {
    /// Create a SIB byte for `[base]` only (no index, scale=0).
    ///
    /// This is needed when the base register encoding is 4 (RSP/R12),
    /// since ModR/M rm=100 signals "SIB follows" instead of [RSP].
    pub fn base_only(base: u8) -> Self {
        Self {
            scale: 0,
            index: 0b100, // no index
            base: base & 0x7,
        }
    }

    /// Create a SIB byte for `[base + index * scale]`.
    ///
    /// `scale_factor` must be 1, 2, 4, or 8 (encoded as 0, 1, 2, 3).
    /// `index` must NOT be RSP (encoding 4) -- RSP means "no index" in SIB.
    pub fn scaled(base: u8, index: u8, scale_factor: u8) -> Self {
        let scale_bits = match scale_factor {
            1 => 0,
            2 => 1,
            4 => 2,
            8 => 3,
            _ => 0, // fallback to scale=1
        };
        Self {
            scale: scale_bits,
            index: index & 0x7,
            base: base & 0x7,
        }
    }

    /// Encode the SIB byte.
    pub fn encode(self) -> u8 {
        (self.scale << 6) | ((self.index & 0x7) << 3) | (self.base & 0x7)
    }
}

// ---------------------------------------------------------------------------
// Operand container for x86-64 instructions
// ---------------------------------------------------------------------------

/// Operands for an x86-64 instruction to be encoded.
///
/// Since the IR `MachInst` is AArch64-typed, the x86-64 encoder uses this
/// separate struct to carry operand information. The ISel or lowering pass
/// populates this before calling the encoder.
#[derive(Debug, Clone)]
pub struct X86InstOperands {
    /// Destination / first source register.
    pub dst: Option<X86PReg>,
    /// Second source register.
    pub src: Option<X86PReg>,
    /// Base register for memory operands.
    pub base: Option<X86PReg>,
    /// Index register for scaled-index (SIB) addressing.
    pub index: Option<X86PReg>,
    /// Scale factor for SIB addressing: 1, 2, 4, or 8.
    pub scale: u8,
    /// Memory displacement/offset.
    pub disp: i64,
    /// Immediate value (sign-extended to 64 bits).
    pub imm: i64,
    /// Condition code (for Jcc).
    pub cc: Option<X86CondCode>,
}

impl X86InstOperands {
    /// Create empty operands.
    pub fn none() -> Self {
        Self {
            dst: None,
            src: None,
            base: None,
            index: None,
            scale: 1,
            disp: 0,
            imm: 0,
            cc: None,
        }
    }

    /// Create operands for a register-register instruction (dst, src).
    pub fn rr(dst: X86PReg, src: X86PReg) -> Self {
        Self {
            dst: Some(dst),
            src: Some(src),
            ..Self::none()
        }
    }

    /// Create operands for a register-immediate instruction (dst, imm).
    pub fn ri(dst: X86PReg, imm: i64) -> Self {
        Self {
            dst: Some(dst),
            imm,
            ..Self::none()
        }
    }

    /// Create operands for a single register operand.
    pub fn r(reg: X86PReg) -> Self {
        Self {
            dst: Some(reg),
            ..Self::none()
        }
    }

    /// Create operands for a register-register-immediate (e.g. IMUL r,r,imm32).
    pub fn rri(dst: X86PReg, src: X86PReg, imm: i64) -> Self {
        Self {
            dst: Some(dst),
            src: Some(src),
            imm,
            ..Self::none()
        }
    }

    /// Create operands for a register-memory instruction (reg, [base+disp]).
    pub fn rm(reg: X86PReg, base: X86PReg, disp: i64) -> Self {
        Self {
            dst: Some(reg),
            base: Some(base),
            disp,
            ..Self::none()
        }
    }

    /// Create operands for a scaled-index memory operand: `[base + index*scale + disp]`.
    ///
    /// `scale` must be 1, 2, 4, or 8.
    pub fn rm_sib(
        reg: X86PReg,
        base: X86PReg,
        index: X86PReg,
        scale: u8,
        disp: i64,
    ) -> Self {
        Self {
            dst: Some(reg),
            base: Some(base),
            index: Some(index),
            scale,
            disp,
            ..Self::none()
        }
    }

    /// Create operands for a RIP-relative LEA: `[RIP + disp32]`.
    pub fn rip_rel(reg: X86PReg, disp: i64) -> Self {
        Self {
            dst: Some(reg),
            disp,
            ..Self::none()
        }
    }

    /// Create operands for a conditional jump (cc, rel32 displacement).
    pub fn jcc(cc: X86CondCode, disp: i64) -> Self {
        Self {
            cc: Some(cc),
            disp,
            ..Self::none()
        }
    }

    /// Create operands for an unconditional jump or call (rel32 displacement).
    pub fn rel(disp: i64) -> Self {
        Self {
            disp,
            ..Self::none()
        }
    }
}

// ---------------------------------------------------------------------------
// X86Encoder — main encoder
// ---------------------------------------------------------------------------

/// x86-64 instruction encoder.
///
/// Encodes `X86Opcode` instructions into machine code bytes.
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

    /// Emit a 16-bit little-endian value.
    pub fn emit_u16_le(&mut self, value: u16) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    /// Emit a 32-bit little-endian value.
    pub fn emit_u32_le(&mut self, value: u32) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    /// Emit a 64-bit little-endian value.
    pub fn emit_u64_le(&mut self, value: u64) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    /// Emit a signed 8-bit immediate.
    pub fn emit_imm8(&mut self, value: i8) {
        self.emit_byte(value as u8);
    }

    /// Emit a signed 32-bit immediate in little-endian.
    pub fn emit_imm32(&mut self, value: i32) {
        self.emit_u32_le(value as u32);
    }

    /// Emit a signed 64-bit immediate in little-endian.
    pub fn emit_imm64(&mut self, value: i64) {
        self.emit_u64_le(value as u64);
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

    /// Emit a SIB byte.
    pub fn emit_sib(&mut self, sib: Sib) {
        self.emit_byte(sib.encode());
    }

    // -----------------------------------------------------------------------
    // Internal encoding helpers
    // -----------------------------------------------------------------------

    /// Build REX prefix for a reg-reg operation with 64-bit operand size.
    /// `reg` goes into ModR/M reg field, `rm` goes into ModR/M rm field.
    fn rex_rr64(reg: X86PReg, rm: X86PReg) -> RexPrefix {
        RexPrefix {
            w: true,
            r: reg.hw_enc() >= 8,
            x: false,
            b: rm.hw_enc() >= 8,
        }
    }

    /// Build REX prefix for a single register operand with 64-bit operand size.
    /// Register goes into ModR/M rm field (opcode extension in reg field).
    fn rex_m64(rm: X86PReg) -> RexPrefix {
        RexPrefix {
            w: true,
            r: false,
            x: false,
            b: rm.hw_enc() >= 8,
        }
    }

    /// Build REX prefix for opcode+rd encoding (register in low 3 bits of opcode).
    /// No REX.W needed for PUSH/POP (default 64-bit in long mode).
    fn rex_oprd(reg: X86PReg, need_w: bool) -> RexPrefix {
        RexPrefix {
            w: need_w,
            r: false,
            x: false,
            b: reg.hw_enc() >= 8,
        }
    }

    /// Encode a reg-reg ALU instruction: `REX.W + opcode /r`.
    ///
    /// ModR/M with mod=11, src in reg field, dst in rm field.
    /// This matches Intel's `/r` encoding where the reg field is the source
    /// for ADD/SUB/AND/OR/XOR/CMP (opcode byte encodes the direction).
    fn encode_alu_rr(&mut self, opcode_byte: u8, dst: X86PReg, src: X86PReg) {
        // For ADD r/m64, r64 (opcode 01): reg=src, rm=dst
        let rex = Self::rex_rr64(src, dst);
        self.emit_rex(rex);
        self.emit_byte(opcode_byte);
        self.emit_modrm(ModRM::reg_reg(src.hw_enc(), dst.hw_enc()));
    }

    /// Encode a reg-imm32 ALU instruction: `REX.W + 81 /ext id`.
    ///
    /// ModR/M with mod=11, opcode extension in reg field, dst in rm field.
    fn encode_alu_ri(&mut self, ext: u8, dst: X86PReg, imm: i32) {
        let rex = Self::rex_m64(dst);
        self.emit_rex(rex);
        self.emit_byte(0x81);
        self.emit_modrm(ModRM::ext_reg(ext, dst.hw_enc()));
        self.emit_imm32(imm);
    }

    /// Encode a unary instruction with opcode extension: `REX.W + opcode /ext`.
    fn encode_unary(&mut self, opcode_byte: u8, ext: u8, reg: X86PReg) {
        let rex = Self::rex_m64(reg);
        self.emit_rex(rex);
        self.emit_byte(opcode_byte);
        self.emit_modrm(ModRM::ext_reg(ext, reg.hw_enc()));
    }

    /// Encode a shift-by-immediate instruction: `REX.W + C1 /ext ib`.
    fn encode_shift_ri(&mut self, ext: u8, dst: X86PReg, imm: i8) {
        let rex = Self::rex_m64(dst);
        self.emit_rex(rex);
        self.emit_byte(0xC1);
        self.emit_modrm(ModRM::ext_reg(ext, dst.hw_enc()));
        self.emit_imm8(imm);
    }

    /// Encode a shift-by-CL instruction: `REX.W + D3 /ext`.
    fn encode_shift_rcl(&mut self, ext: u8, dst: X86PReg) {
        let rex = Self::rex_m64(dst);
        self.emit_rex(rex);
        self.emit_byte(0xD3);
        self.emit_modrm(ModRM::ext_reg(ext, dst.hw_enc()));
    }

    /// Build REX prefix for XMM reg-reg (no REX.W; only need REX.R/REX.B for XMM8-15).
    fn rex_xmm_rr(reg: X86PReg, rm: X86PReg) -> RexPrefix {
        RexPrefix {
            w: false,
            r: reg.hw_enc() >= 8,
            x: false,
            b: rm.hw_enc() >= 8,
        }
    }

    /// Encode an SSE scalar instruction: `[prefix] [REX] 0F opcode /r` (reg-reg, mod=11).
    ///
    /// `prefix` is the mandatory SSE prefix (0xF3 for SS, 0xF2 for SD, 0 for none).
    /// `opcode` is the second byte after 0x0F.
    fn encode_sse_rr(&mut self, prefix: u8, opcode: u8, dst: X86PReg, src: X86PReg) {
        if prefix != 0 {
            self.emit_byte(prefix);
        }
        let rex = Self::rex_xmm_rr(dst, src);
        self.emit_rex(rex);
        self.emit_byte(0x0F);
        self.emit_byte(opcode);
        self.emit_modrm(ModRM::reg_reg(dst.hw_enc(), src.hw_enc()));
    }

    /// Encode an SSE scalar memory load: `[prefix] [REX] 0F opcode /r` (mem operand).
    fn encode_sse_rm(
        &mut self,
        prefix: u8,
        opcode: u8,
        dst: X86PReg,
        base: X86PReg,
        disp: i64,
    ) {
        if prefix != 0 {
            self.emit_byte(prefix);
        }
        let rex = Self::rex_xmm_rr(dst, base);
        self.emit_rex(rex);
        self.emit_byte(0x0F);
        self.emit_byte(opcode);
        self.emit_mem_operand(dst.hw_enc(), base, disp);
    }

    /// Encode an SSE scalar memory store: `[prefix] [REX] 0F opcode /r` (mem operand).
    ///
    /// For stores, the src XMM register goes into the reg field of ModR/M.
    fn encode_sse_mr(
        &mut self,
        prefix: u8,
        opcode: u8,
        src: X86PReg,
        base: X86PReg,
        disp: i64,
    ) {
        if prefix != 0 {
            self.emit_byte(prefix);
        }
        let rex = Self::rex_xmm_rr(src, base);
        self.emit_rex(rex);
        self.emit_byte(0x0F);
        self.emit_byte(opcode);
        self.emit_mem_operand(src.hw_enc(), base, disp);
    }

    /// Encode an SSE scalar RIP-relative load: `[prefix] [REX] 0F opcode ModRM(00 reg 101) disp32`.
    ///
    /// Used for loading float/double constants from a constant pool via
    /// RIP-relative addressing. `disp` is the signed 32-bit displacement
    /// from RIP (the address of the next instruction after this one) to
    /// the constant pool entry.
    fn encode_sse_rip_rel(&mut self, prefix: u8, opcode: u8, dst: X86PReg, disp: i64) {
        if prefix != 0 {
            self.emit_byte(prefix);
        }
        let rex = RexPrefix {
            w: false,
            r: dst.hw_enc() >= 8,
            x: false,
            b: false,
        };
        self.emit_rex(rex);
        self.emit_byte(0x0F);
        self.emit_byte(opcode);
        self.emit_rip_relative(dst.hw_enc(), disp);
    }

    /// Encode a two-byte opcode instruction: `REX.W + 0F + opcode /r` (reg-reg, mod=11).
    ///
    /// Used for CMOV, BSF, BSR, IMUL, etc.
    fn encode_0f_rr64(&mut self, opcode: u8, dst: X86PReg, src: X86PReg) {
        let rex = Self::rex_rr64(dst, src);
        self.emit_rex(rex);
        self.emit_byte(0x0F);
        self.emit_byte(opcode);
        self.emit_modrm(ModRM::reg_reg(dst.hw_enc(), src.hw_enc()));
    }

    /// Emit ModR/M + optional SIB + displacement for a memory operand.
    ///
    /// `reg_or_ext` is the 3-bit value for the ModR/M reg field.
    /// `base` is the base register for addressing.
    /// `disp` is the signed displacement.
    ///
    /// Handles the special cases:
    /// - RSP/R12 (hw_enc & 7 == 4): must use SIB byte
    /// - RBP/R13 (hw_enc & 7 == 5) with disp=0: must use disp8=0
    fn emit_mem_operand(&mut self, reg_or_ext: u8, base: X86PReg, disp: i64) {
        let base_enc = base.hw_enc();
        let base_low3 = base_enc & 0x7;
        let needs_sib = base_low3 == 4; // RSP/R12 encoding

        if disp == 0 && base_low3 != 5 {
            // mod=00: [base] (no displacement)
            if needs_sib {
                self.emit_modrm(ModRM {
                    mode: 0b00,
                    reg: reg_or_ext & 0x7,
                    rm: 0b100, // SIB follows
                });
                self.emit_sib(Sib::base_only(base_enc));
            } else {
                self.emit_modrm(ModRM::indirect(reg_or_ext, base_enc));
            }
        } else if (-128..=127).contains(&disp) {
            // mod=01: [base+disp8]
            if needs_sib {
                self.emit_modrm(ModRM {
                    mode: 0b01,
                    reg: reg_or_ext & 0x7,
                    rm: 0b100,
                });
                self.emit_sib(Sib::base_only(base_enc));
            } else {
                self.emit_modrm(ModRM::indirect_disp8(reg_or_ext, base_enc));
            }
            self.emit_imm8(disp as i8);
        } else {
            // mod=10: [base+disp32]
            if needs_sib {
                self.emit_modrm(ModRM {
                    mode: 0b10,
                    reg: reg_or_ext & 0x7,
                    rm: 0b100,
                });
                self.emit_sib(Sib::base_only(base_enc));
            } else {
                self.emit_modrm(ModRM::indirect_disp32(reg_or_ext, base_enc));
            }
            self.emit_imm32(disp as i32);
        }
    }

    /// Emit ModR/M + SIB + displacement for a scaled-index memory operand.
    ///
    /// Encodes `[base + index * scale + disp]` addressing mode.
    ///
    /// `reg_or_ext` is the 3-bit value for the ModR/M reg field.
    /// `base` is the base register.
    /// `index` is the index register (must NOT be RSP -- hw_enc & 7 == 4).
    /// `scale` is 1, 2, 4, or 8.
    /// `disp` is the signed displacement.
    fn emit_sib_mem_operand(
        &mut self,
        reg_or_ext: u8,
        base: X86PReg,
        index: X86PReg,
        scale: u8,
        disp: i64,
    ) {
        let base_enc = base.hw_enc();
        let base_low3 = base_enc & 0x7;
        let sib = Sib::scaled(base_enc, index.hw_enc(), scale);

        if disp == 0 && base_low3 != 5 {
            // mod=00: [base + index*scale]
            self.emit_modrm(ModRM {
                mode: 0b00,
                reg: reg_or_ext & 0x7,
                rm: 0b100, // SIB follows
            });
            self.emit_sib(sib);
        } else if (-128..=127).contains(&disp) {
            // mod=01: [base + index*scale + disp8]
            self.emit_modrm(ModRM {
                mode: 0b01,
                reg: reg_or_ext & 0x7,
                rm: 0b100,
            });
            self.emit_sib(sib);
            self.emit_imm8(disp as i8);
        } else {
            // mod=10: [base + index*scale + disp32]
            self.emit_modrm(ModRM {
                mode: 0b10,
                reg: reg_or_ext & 0x7,
                rm: 0b100,
            });
            self.emit_sib(sib);
            self.emit_imm32(disp as i32);
        }
    }

    /// Emit ModR/M for RIP-relative addressing: `[RIP + disp32]`.
    ///
    /// Uses ModR/M mod=00, rm=101 which signals RIP-relative in 64-bit mode.
    fn emit_rip_relative(&mut self, reg_or_ext: u8, disp: i64) {
        self.emit_modrm(ModRM {
            mode: 0b00,
            reg: reg_or_ext & 0x7,
            rm: 0b101, // RIP-relative
        });
        self.emit_imm32(disp as i32);
    }

    // -----------------------------------------------------------------------
    // Public encoding API
    // -----------------------------------------------------------------------

    /// Encode a single x86-64 instruction.
    ///
    /// Returns the number of bytes emitted on success.
    pub fn encode_instruction(
        &mut self,
        opcode: X86Opcode,
        ops: &X86InstOperands,
    ) -> Result<usize, X86EncodeError> {
        let start = self.position();

        match opcode {
            // Pseudo-instructions: no encoding
            X86Opcode::Phi | X86Opcode::StackAlloc | X86Opcode::Nop => {
                return Ok(0);
            }

            // =================================================================
            // Multi-byte NOP (hardware encoding for alignment padding)
            // =================================================================
            // NopMulti: 0F 1F /0 variants (2-9 bytes)
            // Reference: Intel SDM Vol 2B, NOP instruction, Table 4-12
            X86Opcode::NopMulti => {
                // Clamp size to [1, 15] bytes. Real callers request 2-9 for
                // a single atomic NOP; we accept a small over-run (up to 15)
                // to cover alignment padding up to a full cache line, then
                // reject anything larger. Without this clamp, a wild `imm`
                // (e.g. `i64::MAX`) coerced through `as usize` would cause
                // `encode_multibyte_nop` to emit gigabytes of bytes and
                // exhaust memory — see #473 (panic-fuzz) for the bug that
                // surfaced this. The previous unbounded recursion path has
                // been converted to iteration in `encode_multibyte_nop`,
                // but we additionally reject pathological `imm` values at
                // the dispatch boundary so adversarial input returns a
                // typed error instead of a giant allocation.
                let requested = if ops.imm > 0 { ops.imm } else { 3 };
                if !(1..=15).contains(&requested) {
                    return Err(X86EncodeError::InvalidOperands(format!(
                        "NopMulti: imm={} out of range [1, 15]",
                        requested
                    )));
                }
                self.encode_multibyte_nop(requested as usize);
            }

            // =================================================================
            // Arithmetic: reg-reg
            // =================================================================
            // ADD r/m64, r64: REX.W + 01 /r
            X86Opcode::AddRR => {
                let (dst, src) = self.require_rr(ops, opcode)?;
                self.encode_alu_rr(0x01, dst, src);
            }
            // SUB r/m64, r64: REX.W + 29 /r
            X86Opcode::SubRR => {
                let (dst, src) = self.require_rr(ops, opcode)?;
                self.encode_alu_rr(0x29, dst, src);
            }
            // AND r/m64, r64: REX.W + 21 /r
            X86Opcode::AndRR => {
                let (dst, src) = self.require_rr(ops, opcode)?;
                self.encode_alu_rr(0x21, dst, src);
            }
            // OR r/m64, r64: REX.W + 09 /r
            X86Opcode::OrRR => {
                let (dst, src) = self.require_rr(ops, opcode)?;
                self.encode_alu_rr(0x09, dst, src);
            }
            // XOR r/m64, r64: REX.W + 31 /r
            X86Opcode::XorRR => {
                let (dst, src) = self.require_rr(ops, opcode)?;
                self.encode_alu_rr(0x31, dst, src);
            }
            // CMP r/m64, r64: REX.W + 39 /r
            X86Opcode::CmpRR => {
                let (dst, src) = self.require_rr(ops, opcode)?;
                self.encode_alu_rr(0x39, dst, src);
            }
            // TEST r/m64, r64: REX.W + 85 /r
            X86Opcode::TestRR => {
                let (dst, src) = self.require_rr(ops, opcode)?;
                self.encode_alu_rr(0x85, dst, src);
            }
            // MOV r/m64, r64: REX.W + 89 /r
            X86Opcode::MovRR => {
                let (dst, src) = self.require_rr(ops, opcode)?;
                self.encode_alu_rr(0x89, dst, src);
            }

            // =================================================================
            // Arithmetic: reg-imm32
            // =================================================================
            // ADD r/m64, imm32: REX.W + 81 /0 id
            X86Opcode::AddRI => {
                let (dst, imm) = self.require_ri(ops, opcode)?;
                self.encode_alu_ri(0, dst, imm);
            }
            // SUB r/m64, imm32: REX.W + 81 /5 id
            X86Opcode::SubRI => {
                let (dst, imm) = self.require_ri(ops, opcode)?;
                self.encode_alu_ri(5, dst, imm);
            }
            // AND r/m64, imm32: REX.W + 81 /4 id
            X86Opcode::AndRI => {
                let (dst, imm) = self.require_ri(ops, opcode)?;
                self.encode_alu_ri(4, dst, imm);
            }
            // OR r/m64, imm32: REX.W + 81 /1 id
            X86Opcode::OrRI => {
                let (dst, imm) = self.require_ri(ops, opcode)?;
                self.encode_alu_ri(1, dst, imm);
            }
            // XOR r/m64, imm32: REX.W + 81 /6 id
            X86Opcode::XorRI => {
                let (dst, imm) = self.require_ri(ops, opcode)?;
                self.encode_alu_ri(6, dst, imm);
            }
            // CMP r/m64, imm32: REX.W + 81 /7 id
            X86Opcode::CmpRI => {
                let (dst, imm) = self.require_ri(ops, opcode)?;
                self.encode_alu_ri(7, dst, imm);
            }
            // CMP r/m64, imm8 (sign-extended): REX.W + 83 /7 ib
            X86Opcode::CmpRI8 => {
                let dst = self.require_dst(ops, opcode)?;
                let imm = ops.imm as i8;
                let rex = Self::rex_m64(dst);
                self.emit_rex(rex);
                self.emit_byte(0x83);
                self.emit_modrm(ModRM::ext_reg(7, dst.hw_enc()));
                self.emit_imm8(imm);
            }
            // TEST r/m64, imm32: REX.W + F7 /0 id
            X86Opcode::TestRI => {
                let dst = self.require_dst(ops, opcode)?;
                let imm = ops.imm as i32;
                let rex = Self::rex_m64(dst);
                self.emit_rex(rex);
                self.emit_byte(0xF7);
                self.emit_modrm(ModRM::ext_reg(0, dst.hw_enc()));
                self.emit_imm32(imm);
            }
            // TEST r64, [base+disp]: REX.W + 85 /r (memory operand form)
            X86Opcode::TestRM => {
                let dst = self.require_dst(ops, opcode)?;
                let base = ops.base.ok_or_else(|| {
                    X86EncodeError::InvalidOperands(format!("{:?}: missing base register", opcode))
                })?;
                let rex = RexPrefix {
                    w: true,
                    r: dst.hw_enc() >= 8,
                    x: false,
                    b: base.hw_enc() >= 8,
                };
                self.emit_rex(rex);
                self.emit_byte(0x85);
                self.emit_mem_operand(dst.hw_enc(), base, ops.disp);
            }

            // =================================================================
            // IMUL
            // =================================================================
            // IMUL r64, r/m64: REX.W + 0F AF /r
            X86Opcode::ImulRR => {
                let (dst, src) = self.require_rr(ops, opcode)?;
                let rex = Self::rex_rr64(dst, src);
                self.emit_rex(rex);
                self.emit_byte(0x0F);
                self.emit_byte(0xAF);
                self.emit_modrm(ModRM::reg_reg(dst.hw_enc(), src.hw_enc()));
            }
            // IMUL r64, r/m64, imm32: REX.W + 69 /r id
            X86Opcode::ImulRRI => {
                let dst = self.require_dst(ops, opcode)?;
                let src = ops.src.ok_or_else(|| {
                    X86EncodeError::InvalidOperands(format!("{:?}: missing src register", opcode))
                })?;
                let imm = ops.imm as i32;
                let rex = Self::rex_rr64(dst, src);
                self.emit_rex(rex);
                self.emit_byte(0x69);
                self.emit_modrm(ModRM::reg_reg(dst.hw_enc(), src.hw_enc()));
                self.emit_imm32(imm);
            }
            // IMUL r64, [base+disp]: REX.W + 0F AF /r (two-operand memory form)
            X86Opcode::ImulRM => {
                let dst = self.require_dst(ops, opcode)?;
                let base = ops.base.ok_or_else(|| {
                    X86EncodeError::InvalidOperands(format!("{:?}: missing base register", opcode))
                })?;
                let rex = RexPrefix {
                    w: true,
                    r: dst.hw_enc() >= 8,
                    x: false,
                    b: base.hw_enc() >= 8,
                };
                self.emit_rex(rex);
                self.emit_byte(0x0F);
                self.emit_byte(0xAF);
                self.emit_mem_operand(dst.hw_enc(), base, ops.disp);
            }

            // =================================================================
            // Unary operations
            // =================================================================
            // NEG r/m64: REX.W + F7 /3
            X86Opcode::Neg => {
                let dst = self.require_dst(ops, opcode)?;
                self.encode_unary(0xF7, 3, dst);
            }
            // NOT r/m64: REX.W + F7 /2
            X86Opcode::Not => {
                let dst = self.require_dst(ops, opcode)?;
                self.encode_unary(0xF7, 2, dst);
            }
            // INC r/m64: REX.W + FF /0
            X86Opcode::Inc => {
                let dst = self.require_dst(ops, opcode)?;
                self.encode_unary(0xFF, 0, dst);
            }
            // DEC r/m64: REX.W + FF /1
            X86Opcode::Dec => {
                let dst = self.require_dst(ops, opcode)?;
                self.encode_unary(0xFF, 1, dst);
            }
            // IDIV r/m64: REX.W + F7 /7
            X86Opcode::Idiv => {
                let dst = self.require_dst(ops, opcode)?;
                self.encode_unary(0xF7, 7, dst);
            }
            // DIV r/m64: REX.W + F7 /6
            X86Opcode::Div => {
                let dst = self.require_dst(ops, opcode)?;
                self.encode_unary(0xF7, 6, dst);
            }
            // CDQ: 99 — sign-extend EAX into EDX:EAX (32-bit, no REX.W)
            X86Opcode::Cdq => {
                self.emit_byte(0x99);
            }
            // CQO: REX.W + 99 — sign-extend RAX into RDX:RAX (64-bit)
            X86Opcode::Cqo => {
                self.emit_byte(0x48); // REX.W
                self.emit_byte(0x99);
            }

            // =================================================================
            // Shifts
            // =================================================================
            // SHL r/m64, imm8: REX.W + C1 /4 ib
            X86Opcode::ShlRI => {
                let dst = self.require_dst(ops, opcode)?;
                self.encode_shift_ri(4, dst, ops.imm as i8);
            }
            // SHR r/m64, imm8: REX.W + C1 /5 ib
            X86Opcode::ShrRI => {
                let dst = self.require_dst(ops, opcode)?;
                self.encode_shift_ri(5, dst, ops.imm as i8);
            }
            // SAR r/m64, imm8: REX.W + C1 /7 ib
            X86Opcode::SarRI => {
                let dst = self.require_dst(ops, opcode)?;
                self.encode_shift_ri(7, dst, ops.imm as i8);
            }
            // SHL r/m64, CL: REX.W + D3 /4
            X86Opcode::ShlRR => {
                let dst = self.require_dst(ops, opcode)?;
                self.encode_shift_rcl(4, dst);
            }
            // SHR r/m64, CL: REX.W + D3 /5
            X86Opcode::ShrRR => {
                let dst = self.require_dst(ops, opcode)?;
                self.encode_shift_rcl(5, dst);
            }
            // SAR r/m64, CL: REX.W + D3 /7
            X86Opcode::SarRR => {
                let dst = self.require_dst(ops, opcode)?;
                self.encode_shift_rcl(7, dst);
            }

            // =================================================================
            // MOV
            // =================================================================
            // MOV r64, imm64: REX.W + B8+rd io (movabs, 10 bytes)
            X86Opcode::MovRI => {
                let dst = self.require_dst(ops, opcode)?;
                let rex = Self::rex_oprd(dst, true);
                self.emit_rex(rex);
                self.emit_byte(0xB8 + (dst.hw_enc() & 0x7));
                self.emit_imm64(ops.imm);
            }
            // MOV r64, [base+disp]: REX.W + 8B /r
            X86Opcode::MovRM => {
                let dst = self.require_dst(ops, opcode)?;
                let base = ops.base.ok_or_else(|| {
                    X86EncodeError::InvalidOperands(format!("{:?}: missing base register", opcode))
                })?;
                let rex = RexPrefix {
                    w: true,
                    r: dst.hw_enc() >= 8,
                    x: false,
                    b: base.hw_enc() >= 8,
                };
                self.emit_rex(rex);
                self.emit_byte(0x8B);
                self.emit_mem_operand(dst.hw_enc(), base, ops.disp);
            }
            // MOV [base+disp], r64: REX.W + 89 /r
            X86Opcode::MovMR => {
                let src = self.require_dst(ops, opcode)?; // dst field holds the src register for stores
                let base = ops.base.ok_or_else(|| {
                    X86EncodeError::InvalidOperands(format!("{:?}: missing base register", opcode))
                })?;
                let rex = RexPrefix {
                    w: true,
                    r: src.hw_enc() >= 8,
                    x: false,
                    b: base.hw_enc() >= 8,
                };
                self.emit_rex(rex);
                self.emit_byte(0x89);
                self.emit_mem_operand(src.hw_enc(), base, ops.disp);
            }

            // AddRM, SubRM, CmpRM: reg-memory forms
            X86Opcode::AddRM => {
                let dst = self.require_dst(ops, opcode)?;
                let base = ops.base.ok_or_else(|| {
                    X86EncodeError::InvalidOperands(format!("{:?}: missing base register", opcode))
                })?;
                let rex = RexPrefix {
                    w: true,
                    r: dst.hw_enc() >= 8,
                    x: false,
                    b: base.hw_enc() >= 8,
                };
                self.emit_rex(rex);
                self.emit_byte(0x03); // ADD r64, r/m64
                self.emit_mem_operand(dst.hw_enc(), base, ops.disp);
            }
            X86Opcode::SubRM => {
                let dst = self.require_dst(ops, opcode)?;
                let base = ops.base.ok_or_else(|| {
                    X86EncodeError::InvalidOperands(format!("{:?}: missing base register", opcode))
                })?;
                let rex = RexPrefix {
                    w: true,
                    r: dst.hw_enc() >= 8,
                    x: false,
                    b: base.hw_enc() >= 8,
                };
                self.emit_rex(rex);
                self.emit_byte(0x2B); // SUB r64, r/m64
                self.emit_mem_operand(dst.hw_enc(), base, ops.disp);
            }
            X86Opcode::CmpRM => {
                let dst = self.require_dst(ops, opcode)?;
                let base = ops.base.ok_or_else(|| {
                    X86EncodeError::InvalidOperands(format!("{:?}: missing base register", opcode))
                })?;
                let rex = RexPrefix {
                    w: true,
                    r: dst.hw_enc() >= 8,
                    x: false,
                    b: base.hw_enc() >= 8,
                };
                self.emit_rex(rex);
                self.emit_byte(0x3B); // CMP r64, r/m64
                self.emit_mem_operand(dst.hw_enc(), base, ops.disp);
            }

            // =================================================================
            // Control flow
            // =================================================================
            // RET: C3
            X86Opcode::Ret => {
                self.emit_byte(0xC3);
            }
            // CALL rel32: E8 cd
            X86Opcode::Call => {
                self.emit_byte(0xE8);
                self.emit_imm32(ops.disp as i32);
            }
            // CALL r64: FF /2
            X86Opcode::CallR => {
                let dst = self.require_dst(ops, opcode)?;
                // CALL r64 does not need REX.W (default 64-bit in long mode),
                // but needs REX.B if the register is R8-R15.
                let rex = Self::rex_oprd(dst, false);
                self.emit_rex(rex);
                self.emit_byte(0xFF);
                self.emit_modrm(ModRM::ext_reg(2, dst.hw_enc()));
            }
            // CALL [base+disp]: FF /2 (indirect call through memory)
            X86Opcode::CallM => {
                let base = ops.base.ok_or_else(|| {
                    X86EncodeError::InvalidOperands(format!("{:?}: missing base register", opcode))
                })?;
                // No REX.W needed (default 64-bit in long mode),
                // but need REX.B if the base register is R8-R15.
                let rex = RexPrefix {
                    w: false,
                    r: false,
                    x: false,
                    b: base.hw_enc() >= 8,
                };
                self.emit_rex(rex);
                self.emit_byte(0xFF);
                self.emit_mem_operand(2, base, ops.disp);
            }
            // JMP rel32: E9 cd
            X86Opcode::Jmp => {
                self.emit_byte(0xE9);
                self.emit_imm32(ops.disp as i32);
            }
            // Jcc rel32: 0F 80+cc cd
            X86Opcode::Jcc => {
                let cc = ops.cc.ok_or_else(|| {
                    X86EncodeError::InvalidOperands("Jcc: missing condition code".into())
                })?;
                self.emit_byte(0x0F);
                self.emit_byte(0x80 + cc.encoding());
                self.emit_imm32(ops.disp as i32);
            }

            // =================================================================
            // Stack
            // =================================================================
            // PUSH r64: 50+rd (no REX.W, only REX.B if R8-R15)
            X86Opcode::Push => {
                let dst = self.require_dst(ops, opcode)?;
                let rex = Self::rex_oprd(dst, false);
                self.emit_rex(rex);
                self.emit_byte(0x50 + (dst.hw_enc() & 0x7));
            }
            // POP r64: 58+rd (no REX.W, only REX.B if R8-R15)
            X86Opcode::Pop => {
                let dst = self.require_dst(ops, opcode)?;
                let rex = Self::rex_oprd(dst, false);
                self.emit_rex(rex);
                self.emit_byte(0x58 + (dst.hw_enc() & 0x7));
            }

            // =================================================================
            // LEA
            // =================================================================
            // LEA r64, [base+disp]: REX.W + 8D /r
            X86Opcode::Lea => {
                let dst = self.require_dst(ops, opcode)?;
                let base = ops.base.ok_or_else(|| {
                    X86EncodeError::InvalidOperands(format!("{:?}: missing base register", opcode))
                })?;
                let rex = RexPrefix {
                    w: true,
                    r: dst.hw_enc() >= 8,
                    x: false,
                    b: base.hw_enc() >= 8,
                };
                self.emit_rex(rex);
                self.emit_byte(0x8D);
                self.emit_mem_operand(dst.hw_enc(), base, ops.disp);
            }
            // LEA r64, [base + index*scale + disp]: REX.W + 8D /r + SIB
            X86Opcode::LeaSib => {
                let dst = self.require_dst(ops, opcode)?;
                let base = ops.base.ok_or_else(|| {
                    X86EncodeError::InvalidOperands(format!("{:?}: missing base register", opcode))
                })?;
                let index = ops.index.ok_or_else(|| {
                    X86EncodeError::InvalidOperands(format!("{:?}: missing index register", opcode))
                })?;
                let rex = RexPrefix {
                    w: true,
                    r: dst.hw_enc() >= 8,
                    x: index.hw_enc() >= 8,
                    b: base.hw_enc() >= 8,
                };
                self.emit_rex(rex);
                self.emit_byte(0x8D);
                self.emit_sib_mem_operand(dst.hw_enc(), base, index, ops.scale, ops.disp);
            }

            // =================================================================
            // LEA RIP-relative
            // =================================================================
            // LEA r64, [RIP+disp32]: REX.W + 8D + ModRM(00 reg 101) + disp32
            X86Opcode::LeaRip => {
                let dst = self.require_dst(ops, opcode)?;
                let rex = RexPrefix {
                    w: true,
                    r: dst.hw_enc() >= 8,
                    x: false,
                    b: false,
                };
                self.emit_rex(rex);
                self.emit_byte(0x8D);
                self.emit_rip_relative(dst.hw_enc(), ops.disp);
            }

            // =================================================================
            // Scaled-index (SIB) memory addressing
            // =================================================================
            // MOV r64, [base+index*scale+disp]: REX.W + 8B /r + SIB
            X86Opcode::MovRMSib => {
                let dst = self.require_dst(ops, opcode)?;
                let base = ops.base.ok_or_else(|| {
                    X86EncodeError::InvalidOperands(format!("{:?}: missing base register", opcode))
                })?;
                let index = ops.index.ok_or_else(|| {
                    X86EncodeError::InvalidOperands(format!("{:?}: missing index register", opcode))
                })?;
                let rex = RexPrefix {
                    w: true,
                    r: dst.hw_enc() >= 8,
                    x: index.hw_enc() >= 8,
                    b: base.hw_enc() >= 8,
                };
                self.emit_rex(rex);
                self.emit_byte(0x8B);
                self.emit_sib_mem_operand(dst.hw_enc(), base, index, ops.scale, ops.disp);
            }
            // MOV [base+index*scale+disp], r64: REX.W + 89 /r + SIB
            X86Opcode::MovMRSib => {
                let src = self.require_dst(ops, opcode)?; // dst field holds src for stores
                let base = ops.base.ok_or_else(|| {
                    X86EncodeError::InvalidOperands(format!("{:?}: missing base register", opcode))
                })?;
                let index = ops.index.ok_or_else(|| {
                    X86EncodeError::InvalidOperands(format!("{:?}: missing index register", opcode))
                })?;
                let rex = RexPrefix {
                    w: true,
                    r: src.hw_enc() >= 8,
                    x: index.hw_enc() >= 8,
                    b: base.hw_enc() >= 8,
                };
                self.emit_rex(rex);
                self.emit_byte(0x89);
                self.emit_sib_mem_operand(src.hw_enc(), base, index, ops.scale, ops.disp);
            }

            // =================================================================
            // MOVZX / MOVSX
            // =================================================================
            // MOVZX r64, r/m8: REX.W + 0F B6 /r (mod=11 for reg-reg)
            // MOVZX r64, r/m16: REX.W + 0F B7 /r
            // We encode the r/m8 form by default (8->64 zero-extend).
            // The ISel should pick between B6 (byte) and B7 (word) variants.
            X86Opcode::Movzx => {
                let (dst, src) = self.require_rr(ops, opcode)?;
                let rex = Self::rex_rr64(dst, src);
                self.emit_rex(rex);
                self.emit_byte(0x0F);
                // Use B6 for 8-bit source, B7 for 16-bit. Default to B6.
                self.emit_byte(0xB6);
                self.emit_modrm(ModRM::reg_reg(dst.hw_enc(), src.hw_enc()));
            }
            // MOVSXD r64, r/m32: REX.W + 63 /r (sign-extend 32->64)
            X86Opcode::Movsx => {
                let (dst, src) = self.require_rr(ops, opcode)?;
                let rex = Self::rex_rr64(dst, src);
                self.emit_rex(rex);
                self.emit_byte(0x63);
                self.emit_modrm(ModRM::reg_reg(dst.hw_enc(), src.hw_enc()));
            }
            // MOVZX r64, r/m16: REX.W + 0F B7 /r (zero-extend word to qword)
            X86Opcode::MovzxW => {
                let (dst, src) = self.require_rr(ops, opcode)?;
                self.encode_0f_rr64(0xB7, dst, src);
            }
            // MOVSX r64, r/m8: REX.W + 0F BE /r (sign-extend byte to qword)
            X86Opcode::MovsxB => {
                let (dst, src) = self.require_rr(ops, opcode)?;
                self.encode_0f_rr64(0xBE, dst, src);
            }
            // MOVSX r64, r/m16: REX.W + 0F BF /r (sign-extend word to qword)
            X86Opcode::MovsxW => {
                let (dst, src) = self.require_rr(ops, opcode)?;
                self.encode_0f_rr64(0xBF, dst, src);
            }

            // =================================================================
            // SSE scalar double-precision (F2 0F prefix)
            // =================================================================
            // ADDSD xmm, xmm: F2 0F 58 /r
            X86Opcode::Addsd => {
                let (dst, src) = self.require_rr(ops, opcode)?;
                self.encode_sse_rr(0xF2, 0x58, dst, src);
            }
            // SUBSD xmm, xmm: F2 0F 5C /r
            X86Opcode::Subsd => {
                let (dst, src) = self.require_rr(ops, opcode)?;
                self.encode_sse_rr(0xF2, 0x5C, dst, src);
            }
            // MULSD xmm, xmm: F2 0F 59 /r
            X86Opcode::Mulsd => {
                let (dst, src) = self.require_rr(ops, opcode)?;
                self.encode_sse_rr(0xF2, 0x59, dst, src);
            }
            // DIVSD xmm, xmm: F2 0F 5E /r
            X86Opcode::Divsd => {
                let (dst, src) = self.require_rr(ops, opcode)?;
                self.encode_sse_rr(0xF2, 0x5E, dst, src);
            }
            // MOVSD xmm, xmm: F2 0F 10 /r
            X86Opcode::MovsdRR => {
                let (dst, src) = self.require_rr(ops, opcode)?;
                self.encode_sse_rr(0xF2, 0x10, dst, src);
            }
            // MOVSD xmm, [mem]: F2 0F 10 /r
            X86Opcode::MovsdRM => {
                let dst = self.require_dst(ops, opcode)?;
                let base = ops.base.ok_or_else(|| {
                    X86EncodeError::InvalidOperands(format!("{:?}: missing base register", opcode))
                })?;
                self.encode_sse_rm(0xF2, 0x10, dst, base, ops.disp);
            }
            // MOVSD [mem], xmm: F2 0F 11 /r
            X86Opcode::MovsdMR => {
                let src = self.require_dst(ops, opcode)?; // dst field holds src for stores
                let base = ops.base.ok_or_else(|| {
                    X86EncodeError::InvalidOperands(format!("{:?}: missing base register", opcode))
                })?;
                self.encode_sse_mr(0xF2, 0x11, src, base, ops.disp);
            }
            // UCOMISD xmm, xmm: 66 0F 2E /r
            X86Opcode::Ucomisd => {
                let (dst, src) = self.require_rr(ops, opcode)?;
                self.encode_sse_rr(0x66, 0x2E, dst, src);
            }

            // =================================================================
            // SSE scalar single-precision (F3 0F prefix)
            // =================================================================
            // ADDSS xmm, xmm: F3 0F 58 /r
            X86Opcode::Addss => {
                let (dst, src) = self.require_rr(ops, opcode)?;
                self.encode_sse_rr(0xF3, 0x58, dst, src);
            }
            // SUBSS xmm, xmm: F3 0F 5C /r
            X86Opcode::Subss => {
                let (dst, src) = self.require_rr(ops, opcode)?;
                self.encode_sse_rr(0xF3, 0x5C, dst, src);
            }
            // MULSS xmm, xmm: F3 0F 59 /r
            X86Opcode::Mulss => {
                let (dst, src) = self.require_rr(ops, opcode)?;
                self.encode_sse_rr(0xF3, 0x59, dst, src);
            }
            // DIVSS xmm, xmm: F3 0F 5E /r
            X86Opcode::Divss => {
                let (dst, src) = self.require_rr(ops, opcode)?;
                self.encode_sse_rr(0xF3, 0x5E, dst, src);
            }
            // MOVSS xmm, xmm: F3 0F 10 /r
            X86Opcode::MovssRR => {
                let (dst, src) = self.require_rr(ops, opcode)?;
                self.encode_sse_rr(0xF3, 0x10, dst, src);
            }
            // MOVSS xmm, [mem]: F3 0F 10 /r
            X86Opcode::MovssRM => {
                let dst = self.require_dst(ops, opcode)?;
                let base = ops.base.ok_or_else(|| {
                    X86EncodeError::InvalidOperands(format!("{:?}: missing base register", opcode))
                })?;
                self.encode_sse_rm(0xF3, 0x10, dst, base, ops.disp);
            }
            // MOVSS [mem], xmm: F3 0F 11 /r
            X86Opcode::MovssMR => {
                let src = self.require_dst(ops, opcode)?;
                let base = ops.base.ok_or_else(|| {
                    X86EncodeError::InvalidOperands(format!("{:?}: missing base register", opcode))
                })?;
                self.encode_sse_mr(0xF3, 0x11, src, base, ops.disp);
            }
            // UCOMISS xmm, xmm: 0F 2E /r (no prefix)
            X86Opcode::Ucomiss => {
                let (dst, src) = self.require_rr(ops, opcode)?;
                self.encode_sse_rr(0, 0x2E, dst, src);
            }

            // =================================================================
            // SSE RIP-relative constant pool loads
            // =================================================================
            // MOVSS xmm, [RIP+disp32]: F3 [REX] 0F 10 ModRM(00 reg 101) disp32
            X86Opcode::MovssRipRel => {
                let dst = self.require_dst(ops, opcode)?;
                self.encode_sse_rip_rel(0xF3, 0x10, dst, ops.disp);
            }
            // MOVSD xmm, [RIP+disp32]: F2 [REX] 0F 10 ModRM(00 reg 101) disp32
            X86Opcode::MovsdRipRel => {
                let dst = self.require_dst(ops, opcode)?;
                self.encode_sse_rip_rel(0xF2, 0x10, dst, ops.disp);
            }

            // =================================================================
            // SSE type conversion
            // =================================================================
            // CVTSI2SD xmm, r64: F2 REX.W 0F 2A /r
            X86Opcode::Cvtsi2sd => {
                let (dst, src) = self.require_rr(ops, opcode)?;
                self.emit_byte(0xF2);
                let rex = RexPrefix {
                    w: true,
                    r: dst.hw_enc() >= 8,
                    x: false,
                    b: src.hw_enc() >= 8,
                };
                self.emit_rex(rex);
                self.emit_byte(0x0F);
                self.emit_byte(0x2A);
                self.emit_modrm(ModRM::reg_reg(dst.hw_enc(), src.hw_enc()));
            }
            // CVTSD2SI r64, xmm: F2 REX.W 0F 2D /r (truncate)
            X86Opcode::Cvtsd2si => {
                let (dst, src) = self.require_rr(ops, opcode)?;
                self.emit_byte(0xF2);
                let rex = RexPrefix {
                    w: true,
                    r: dst.hw_enc() >= 8,
                    x: false,
                    b: src.hw_enc() >= 8,
                };
                self.emit_rex(rex);
                self.emit_byte(0x0F);
                self.emit_byte(0x2D);
                self.emit_modrm(ModRM::reg_reg(dst.hw_enc(), src.hw_enc()));
            }
            // CVTSI2SS xmm, r64: F3 REX.W 0F 2A /r
            X86Opcode::Cvtsi2ss => {
                let (dst, src) = self.require_rr(ops, opcode)?;
                self.emit_byte(0xF3);
                let rex = RexPrefix {
                    w: true,
                    r: dst.hw_enc() >= 8,
                    x: false,
                    b: src.hw_enc() >= 8,
                };
                self.emit_rex(rex);
                self.emit_byte(0x0F);
                self.emit_byte(0x2A);
                self.emit_modrm(ModRM::reg_reg(dst.hw_enc(), src.hw_enc()));
            }
            // CVTSS2SI r64, xmm: F3 REX.W 0F 2D /r (truncate)
            X86Opcode::Cvtss2si => {
                let (dst, src) = self.require_rr(ops, opcode)?;
                self.emit_byte(0xF3);
                let rex = RexPrefix {
                    w: true,
                    r: dst.hw_enc() >= 8,
                    x: false,
                    b: src.hw_enc() >= 8,
                };
                self.emit_rex(rex);
                self.emit_byte(0x0F);
                self.emit_byte(0x2D);
                self.emit_modrm(ModRM::reg_reg(dst.hw_enc(), src.hw_enc()));
            }
            // CVTSD2SS xmm, xmm: F2 0F 5A /r
            X86Opcode::Cvtsd2ss => {
                let (dst, src) = self.require_rr(ops, opcode)?;
                self.encode_sse_rr(0xF2, 0x5A, dst, src);
            }
            // CVTSS2SD xmm, xmm: F3 0F 5A /r
            X86Opcode::Cvtss2sd => {
                let (dst, src) = self.require_rr(ops, opcode)?;
                self.encode_sse_rr(0xF3, 0x5A, dst, src);
            }

            // =================================================================
            // CMOVcc — conditional move
            // =================================================================
            // CMOVcc r64, r64: REX.W + 0F 40+cc /r
            X86Opcode::Cmovcc => {
                let (dst, src) = self.require_rr(ops, opcode)?;
                let cc = ops.cc.ok_or_else(|| {
                    X86EncodeError::InvalidOperands("CMOVcc: missing condition code".into())
                })?;
                let rex = Self::rex_rr64(dst, src);
                self.emit_rex(rex);
                self.emit_byte(0x0F);
                self.emit_byte(0x40 + cc.encoding());
                self.emit_modrm(ModRM::reg_reg(dst.hw_enc(), src.hw_enc()));
            }

            // =================================================================
            // SETcc — set byte on condition
            // =================================================================
            // SETcc r/m8: 0F 90+cc /0 (ModR/M mod=11, reg=0, rm=reg)
            // Needs REX prefix if dst is R8-R15 or SPL/BPL/SIL/DIL.
            X86Opcode::Setcc => {
                let dst = self.require_dst(ops, opcode)?;
                let cc = ops.cc.ok_or_else(|| {
                    X86EncodeError::InvalidOperands("SETcc: missing condition code".into())
                })?;
                // SETcc operates on 8-bit register, no REX.W.
                // Need REX.B if destination encoding >= 8.
                let rex = RexPrefix {
                    w: false,
                    r: false,
                    x: false,
                    b: dst.hw_enc() >= 8,
                };
                self.emit_rex(rex);
                self.emit_byte(0x0F);
                self.emit_byte(0x90 + cc.encoding());
                self.emit_modrm(ModRM::ext_reg(0, dst.hw_enc()));
            }

            // =================================================================
            // Bit manipulation
            // =================================================================
            // BSF r64, r64: REX.W + 0F BC /r
            X86Opcode::Bsf => {
                let (dst, src) = self.require_rr(ops, opcode)?;
                self.encode_0f_rr64(0xBC, dst, src);
            }
            // BSR r64, r64: REX.W + 0F BD /r
            X86Opcode::Bsr => {
                let (dst, src) = self.require_rr(ops, opcode)?;
                self.encode_0f_rr64(0xBD, dst, src);
            }
            // TZCNT r64, r64: F3 REX.W + 0F BC /r (rep-prefixed BSF)
            X86Opcode::Tzcnt => {
                let (dst, src) = self.require_rr(ops, opcode)?;
                self.emit_byte(0xF3); // REP prefix
                let rex = Self::rex_rr64(dst, src);
                self.emit_rex(rex);
                self.emit_byte(0x0F);
                self.emit_byte(0xBC);
                self.emit_modrm(ModRM::reg_reg(dst.hw_enc(), src.hw_enc()));
            }
            // LZCNT r64, r64: F3 REX.W + 0F BD /r (rep-prefixed BSR)
            X86Opcode::Lzcnt => {
                let (dst, src) = self.require_rr(ops, opcode)?;
                self.emit_byte(0xF3); // REP prefix
                let rex = Self::rex_rr64(dst, src);
                self.emit_rex(rex);
                self.emit_byte(0x0F);
                self.emit_byte(0xBD);
                self.emit_modrm(ModRM::reg_reg(dst.hw_enc(), src.hw_enc()));
            }
            // POPCNT r64, r64: F3 REX.W + 0F B8 /r
            X86Opcode::Popcnt => {
                let (dst, src) = self.require_rr(ops, opcode)?;
                self.emit_byte(0xF3); // REP prefix
                let rex = Self::rex_rr64(dst, src);
                self.emit_rex(rex);
                self.emit_byte(0x0F);
                self.emit_byte(0xB8);
                self.emit_modrm(ModRM::reg_reg(dst.hw_enc(), src.hw_enc()));
            }
            // BT r/m64, imm8: REX.W + 0F BA /4 ib
            X86Opcode::BtRI => {
                let dst = self.require_dst(ops, opcode)?;
                let rex = Self::rex_m64(dst);
                self.emit_rex(rex);
                self.emit_byte(0x0F);
                self.emit_byte(0xBA);
                self.emit_modrm(ModRM::ext_reg(4, dst.hw_enc()));
                self.emit_imm8(ops.imm as i8);
            }
            // BSWAP r64: REX.W + 0F C8+rd
            X86Opcode::Bswap => {
                let dst = self.require_dst(ops, opcode)?;
                let rex = Self::rex_oprd(dst, true);
                self.emit_rex(rex);
                self.emit_byte(0x0F);
                self.emit_byte(0xC8 + (dst.hw_enc() & 0x7));
            }

            // =================================================================
            // Atomic / exchange
            // =================================================================
            // XCHG r/m64, r64: REX.W + 87 /r
            X86Opcode::Xchg => {
                let (dst, src) = self.require_rr(ops, opcode)?;
                self.encode_alu_rr(0x87, dst, src);
            }
            // LOCK CMPXCHG r/m64, r64: F0 + REX.W + 0F B1 /r
            // Compare RAX with r/m64; if equal, ZF is set and r64 is
            // stored into r/m64. Otherwise, r/m64 is loaded into RAX.
            X86Opcode::Cmpxchg => {
                let (dst, src) = self.require_rr(ops, opcode)?;
                self.emit_byte(0xF0); // LOCK prefix
                let rex = Self::rex_rr64(src, dst);
                self.emit_rex(rex);
                self.emit_byte(0x0F);
                self.emit_byte(0xB1);
                self.emit_modrm(ModRM::reg_reg(src.hw_enc(), dst.hw_enc()));
            }

            // =================================================================
            // GPR <-> XMM transfers
            // =================================================================
            // MOVD xmm, r/m32: 66 0F 6E /r (no REX.W, 32-bit)
            X86Opcode::MovdToXmm => {
                let (dst, src) = self.require_rr(ops, opcode)?;
                self.emit_byte(0x66);
                let rex = Self::rex_xmm_rr(dst, src);
                self.emit_rex(rex);
                self.emit_byte(0x0F);
                self.emit_byte(0x6E);
                self.emit_modrm(ModRM::reg_reg(dst.hw_enc(), src.hw_enc()));
            }
            // MOVD r/m32, xmm: 66 0F 7E /r (no REX.W, 32-bit)
            X86Opcode::MovdFromXmm => {
                let (dst, src) = self.require_rr(ops, opcode)?;
                // dst=GPR (in rm), src=XMM (in reg)
                self.emit_byte(0x66);
                let rex = Self::rex_xmm_rr(src, dst);
                self.emit_rex(rex);
                self.emit_byte(0x0F);
                self.emit_byte(0x7E);
                self.emit_modrm(ModRM::reg_reg(src.hw_enc(), dst.hw_enc()));
            }
            // MOVQ xmm, r/m64: 66 REX.W 0F 6E /r (64-bit with REX.W)
            X86Opcode::MovqToXmm => {
                let (dst, src) = self.require_rr(ops, opcode)?;
                self.emit_byte(0x66);
                let rex = RexPrefix {
                    w: true,
                    r: dst.hw_enc() >= 8,
                    x: false,
                    b: src.hw_enc() >= 8,
                };
                self.emit_rex(rex);
                self.emit_byte(0x0F);
                self.emit_byte(0x6E);
                self.emit_modrm(ModRM::reg_reg(dst.hw_enc(), src.hw_enc()));
            }
            // MOVQ r/m64, xmm: 66 REX.W 0F 7E /r (64-bit with REX.W)
            X86Opcode::MovqFromXmm => {
                let (dst, src) = self.require_rr(ops, opcode)?;
                // dst=GPR (in rm), src=XMM (in reg)
                self.emit_byte(0x66);
                let rex = RexPrefix {
                    w: true,
                    r: src.hw_enc() >= 8,
                    x: false,
                    b: dst.hw_enc() >= 8,
                };
                self.emit_rex(rex);
                self.emit_byte(0x0F);
                self.emit_byte(0x7E);
                self.emit_modrm(ModRM::reg_reg(src.hw_enc(), dst.hw_enc()));
            }
        }

        Ok(self.position() - start)
    }

    // -----------------------------------------------------------------------
    // Operand extraction helpers
    // -----------------------------------------------------------------------

    fn require_dst(
        &self,
        ops: &X86InstOperands,
        opcode: X86Opcode,
    ) -> Result<X86PReg, X86EncodeError> {
        ops.dst.ok_or_else(|| {
            X86EncodeError::InvalidOperands(format!("{:?}: missing dst register", opcode))
        })
    }

    fn require_rr(
        &self,
        ops: &X86InstOperands,
        opcode: X86Opcode,
    ) -> Result<(X86PReg, X86PReg), X86EncodeError> {
        let dst = ops.dst.ok_or_else(|| {
            X86EncodeError::InvalidOperands(format!("{:?}: missing dst register", opcode))
        })?;
        let src = ops.src.ok_or_else(|| {
            X86EncodeError::InvalidOperands(format!("{:?}: missing src register", opcode))
        })?;
        Ok((dst, src))
    }

    fn require_ri(
        &self,
        ops: &X86InstOperands,
        opcode: X86Opcode,
    ) -> Result<(X86PReg, i32), X86EncodeError> {
        let dst = ops.dst.ok_or_else(|| {
            X86EncodeError::InvalidOperands(format!("{:?}: missing dst register", opcode))
        })?;
        Ok((dst, ops.imm as i32))
    }

    /// Encode a hardware NOP instruction (0x90).
    ///
    /// Note: `X86Opcode::Nop` is a pseudo with no encoding. Call this
    /// directly when you need a real 1-byte NOP in the output stream.
    pub fn encode_nop(&mut self) {
        self.emit_byte(0x90);
    }

    /// Encode a multi-byte NOP of the given size (0-9 bytes).
    ///
    /// Reference: Intel SDM Vol 2B, NOP instruction, Table 4-12.
    /// Recommended multi-byte NOP sequences for each length:
    /// - 1 byte: 90
    /// - 2 bytes: 66 90
    /// - 3 bytes: 0F 1F 00
    /// - 4 bytes: 0F 1F 40 00
    /// - 5 bytes: 0F 1F 44 00 00
    /// - 6 bytes: 66 0F 1F 44 00 00
    /// - 7 bytes: 0F 1F 80 00 00 00 00
    /// - 8 bytes: 0F 1F 84 00 00 00 00 00
    /// - 9 bytes: 66 0F 1F 84 00 00 00 00 00
    ///
    /// For sizes > 9, emits multiple 9-byte NOP sequences iteratively. The
    /// original implementation recursed via `encode_multibyte_nop(size - 9)`,
    /// which overflowed the stack for adversarial callers (e.g. a wild
    /// `ops.imm` of `i64::MAX` coerced through `NopMulti`). The panic-fuzz
    /// harness `panic_fuzz_encode_x86_64.rs` (#473) surfaced this as a real
    /// SIGABRT on macOS aarch64; converting to iteration removes the
    /// unbounded-recursion vector without changing the emitted byte sequence
    /// for any `size`.
    pub fn encode_multibyte_nop(&mut self, size: usize) {
        // Emit 9-byte NOP sequences until the remaining size fits in one
        // atomic NOP emission. This preserves bit-identical output with the
        // previous recursive implementation but bounds stack depth to O(1).
        let mut remaining = size;
        while remaining > 9 {
            // 9 bytes: 66 NOP DWORD ptr [RAX + RAX*1 + 00000000]
            self.emit_byte(0x66);
            self.emit_byte(0x0F);
            self.emit_byte(0x1F);
            self.emit_byte(0x84);
            self.emit_byte(0x00);
            self.emit_byte(0x00);
            self.emit_byte(0x00);
            self.emit_byte(0x00);
            self.emit_byte(0x00);
            remaining -= 9;
        }
        match remaining {
            0 => {}
            1 => {
                self.emit_byte(0x90);
            }
            2 => {
                self.emit_byte(0x66);
                self.emit_byte(0x90);
            }
            3 => {
                // NOP DWORD ptr [RAX]
                self.emit_byte(0x0F);
                self.emit_byte(0x1F);
                self.emit_byte(0x00);
            }
            4 => {
                // NOP DWORD ptr [RAX + 00]
                self.emit_byte(0x0F);
                self.emit_byte(0x1F);
                self.emit_byte(0x40);
                self.emit_byte(0x00);
            }
            5 => {
                // NOP DWORD ptr [RAX + RAX*1 + 00]
                self.emit_byte(0x0F);
                self.emit_byte(0x1F);
                self.emit_byte(0x44);
                self.emit_byte(0x00);
                self.emit_byte(0x00);
            }
            6 => {
                // 66 NOP DWORD ptr [RAX + RAX*1 + 00]
                self.emit_byte(0x66);
                self.emit_byte(0x0F);
                self.emit_byte(0x1F);
                self.emit_byte(0x44);
                self.emit_byte(0x00);
                self.emit_byte(0x00);
            }
            7 => {
                // NOP DWORD ptr [RAX + 00000000]
                self.emit_byte(0x0F);
                self.emit_byte(0x1F);
                self.emit_byte(0x80);
                self.emit_byte(0x00);
                self.emit_byte(0x00);
                self.emit_byte(0x00);
                self.emit_byte(0x00);
            }
            8 => {
                // NOP DWORD ptr [RAX + RAX*1 + 00000000]
                self.emit_byte(0x0F);
                self.emit_byte(0x1F);
                self.emit_byte(0x84);
                self.emit_byte(0x00);
                self.emit_byte(0x00);
                self.emit_byte(0x00);
                self.emit_byte(0x00);
                self.emit_byte(0x00);
            }
            _ => {
                // Unreachable under the loop above (remaining is in 0..=9
                // after the while-loop exits). Kept as a defensive 9-byte
                // emission to preserve behaviour if remaining ever somehow
                // equals exactly 9.
                self.emit_byte(0x66);
                self.emit_byte(0x0F);
                self.emit_byte(0x1F);
                self.emit_byte(0x84);
                self.emit_byte(0x00);
                self.emit_byte(0x00);
                self.emit_byte(0x00);
                self.emit_byte(0x00);
                self.emit_byte(0x00);
            }
        }
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
    use llvm2_ir::x86_64_ops::{X86CondCode, X86Opcode};
    use llvm2_ir::x86_64_regs::{
        R8, R9, R10, R11, R12, R13, R14, R15, RAX, RBP, RBX, RCX, RDI, RDX, RSI, RSP,
        AL, CL, R8B,
        AX, CX, R14W,
        XMM0, XMM1, XMM8, XMM15,
    };

    // Helper to encode an instruction and return the bytes.
    fn encode(opcode: X86Opcode, ops: &X86InstOperands) -> Vec<u8> {
        let mut enc = X86Encoder::new();
        enc.encode_instruction(opcode, ops).unwrap();
        enc.finish()
    }

    // -----------------------------------------------------------------------
    // REX prefix tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_rex_prefix_not_needed() {
        let rex = RexPrefix::default();
        assert!(!rex.is_needed());
    }

    #[test]
    fn test_rex_prefix_w() {
        let rex = RexPrefix {
            w: true,
            ..Default::default()
        };
        assert!(rex.is_needed());
        assert_eq!(rex.encode(), 0x48);
    }

    #[test]
    fn test_rex_prefix_all() {
        let rex = RexPrefix {
            w: true,
            r: true,
            x: true,
            b: true,
        };
        assert_eq!(rex.encode(), 0x4F);
    }

    #[test]
    fn test_rex_prefix_b_only() {
        let rex = RexPrefix {
            b: true,
            ..Default::default()
        };
        assert_eq!(rex.encode(), 0x41);
    }

    // -----------------------------------------------------------------------
    // ModR/M tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_modrm_reg_reg() {
        // MOV RAX, RBX: mod=11, reg=RAX(0), rm=RBX(3)
        let modrm = ModRM::reg_reg(0, 3);
        assert_eq!(modrm.encode(), 0b11_000_011);
    }

    #[test]
    fn test_modrm_encode() {
        // mod=10, reg=5, rm=4 (SIB follows)
        let modrm = ModRM {
            mode: 0b10,
            reg: 5,
            rm: 4,
        };
        assert_eq!(modrm.encode(), 0b10_101_100);
    }

    // -----------------------------------------------------------------------
    // SIB tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_sib_base_only_rsp() {
        // SIB for [RSP]: scale=0, index=none(4), base=RSP(4)
        let sib = Sib::base_only(4);
        assert_eq!(sib.encode(), 0b00_100_100);
    }

    #[test]
    fn test_sib_base_only_r12() {
        // SIB for [R12]: same low 3 bits as RSP
        let sib = Sib::base_only(12); // 12 & 7 = 4
        assert_eq!(sib.encode(), 0b00_100_100);
    }

    // -----------------------------------------------------------------------
    // Encoder basic tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_encoder_new() {
        let enc = X86Encoder::new();
        assert_eq!(enc.position(), 0);
        assert!(enc.bytes.is_empty());
    }

    #[test]
    fn test_encoder_emit_bytes() {
        let mut enc = X86Encoder::new();
        enc.emit_byte(0x90);
        enc.emit_byte(0xC3);
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
        assert_eq!(
            enc.finish(),
            vec![0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01]
        );
    }

    // -----------------------------------------------------------------------
    // Pseudo-instruction tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_encode_pseudo_succeeds() {
        let ops = X86InstOperands::none();
        let empty: Vec<u8> = vec![];
        assert_eq!(encode(X86Opcode::Phi, &ops), empty);
        assert_eq!(encode(X86Opcode::Nop, &ops), empty);
        assert_eq!(encode(X86Opcode::StackAlloc, &ops), empty);
    }

    // -----------------------------------------------------------------------
    // NOP (hardware)
    // -----------------------------------------------------------------------

    #[test]
    fn test_encode_nop() {
        let mut enc = X86Encoder::new();
        enc.encode_nop();
        assert_eq!(enc.finish(), vec![0x90]);
    }

    // -----------------------------------------------------------------------
    // ADD tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_add_rax_rcx() {
        // ADD RAX, RCX: REX.W(48) + 01 + ModRM(11 001 000) = C8
        let bytes = encode(X86Opcode::AddRR, &X86InstOperands::rr(RAX, RCX));
        assert_eq!(bytes, vec![0x48, 0x01, 0xC8]);
    }

    #[test]
    fn test_add_rbx_rdx() {
        // ADD RBX, RDX: REX.W(48) + 01 + ModRM(11 010 011) = D3
        let bytes = encode(X86Opcode::AddRR, &X86InstOperands::rr(RBX, RDX));
        assert_eq!(bytes, vec![0x48, 0x01, 0xD3]);
    }

    #[test]
    fn test_add_r8_r9() {
        // ADD R8, R9: REX.WRB(4D) + 01 + ModRM(11 001 000) = C8
        // src=R9(hw=9, bit3=1 -> REX.R), dst=R8(hw=8, bit3=1 -> REX.B)
        let bytes = encode(X86Opcode::AddRR, &X86InstOperands::rr(R8, R9));
        assert_eq!(bytes, vec![0x4D, 0x01, 0xC8]);
    }

    #[test]
    fn test_add_rax_r8() {
        // ADD RAX, R8: src=R8(hw=8, bit3=1 -> REX.R), dst=RAX(hw=0)
        // REX.WR(4C) + 01 + ModRM(11 000 000) = C0
        let bytes = encode(X86Opcode::AddRR, &X86InstOperands::rr(RAX, R8));
        assert_eq!(bytes, vec![0x4C, 0x01, 0xC0]);
    }

    #[test]
    fn test_add_r15_rax() {
        // ADD R15, RAX: src=RAX(hw=0), dst=R15(hw=15, bit3=1 -> REX.B)
        // REX.WB(49) + 01 + ModRM(11 000 111) = C7
        let bytes = encode(X86Opcode::AddRR, &X86InstOperands::rr(R15, RAX));
        assert_eq!(bytes, vec![0x49, 0x01, 0xC7]);
    }

    #[test]
    fn test_add_rax_imm32() {
        // ADD RAX, 42: REX.W(48) + 81 + ModRM(11 000 000) + imm32(2A000000)
        let bytes = encode(X86Opcode::AddRI, &X86InstOperands::ri(RAX, 42));
        assert_eq!(bytes, vec![0x48, 0x81, 0xC0, 0x2A, 0x00, 0x00, 0x00]);
    }

    #[test]
    fn test_add_r12_imm32() {
        // ADD R12, 100: REX.WB(49) + 81 + ModRM(11 000 100) + imm32
        let bytes = encode(X86Opcode::AddRI, &X86InstOperands::ri(R12, 100));
        assert_eq!(bytes, vec![0x49, 0x81, 0xC4, 0x64, 0x00, 0x00, 0x00]);
    }

    // -----------------------------------------------------------------------
    // SUB tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_sub_rax_rcx() {
        // SUB RAX, RCX: REX.W(48) + 29 + ModRM(11 001 000) = C8
        let bytes = encode(X86Opcode::SubRR, &X86InstOperands::rr(RAX, RCX));
        assert_eq!(bytes, vec![0x48, 0x29, 0xC8]);
    }

    #[test]
    fn test_sub_rax_imm32() {
        // SUB RAX, 10: REX.W(48) + 81 + ModRM(11 101 000) = E8 + imm32
        let bytes = encode(X86Opcode::SubRI, &X86InstOperands::ri(RAX, 10));
        assert_eq!(bytes, vec![0x48, 0x81, 0xE8, 0x0A, 0x00, 0x00, 0x00]);
    }

    // -----------------------------------------------------------------------
    // AND/OR/XOR tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_and_rax_rcx() {
        // AND RAX, RCX: REX.W + 21 + ModRM(11 001 000)
        let bytes = encode(X86Opcode::AndRR, &X86InstOperands::rr(RAX, RCX));
        assert_eq!(bytes, vec![0x48, 0x21, 0xC8]);
    }

    #[test]
    fn test_or_rax_rdx() {
        // OR RAX, RDX: REX.W + 09 + ModRM(11 010 000)
        let bytes = encode(X86Opcode::OrRR, &X86InstOperands::rr(RAX, RDX));
        assert_eq!(bytes, vec![0x48, 0x09, 0xD0]);
    }

    #[test]
    fn test_xor_rax_rax() {
        // XOR RAX, RAX: REX.W + 31 + ModRM(11 000 000) = C0
        let bytes = encode(X86Opcode::XorRR, &X86InstOperands::rr(RAX, RAX));
        assert_eq!(bytes, vec![0x48, 0x31, 0xC0]);
    }

    #[test]
    fn test_and_rcx_imm32() {
        // AND RCX, 0xFF: REX.W + 81 + ModRM(11 100 001) + imm32
        let bytes = encode(X86Opcode::AndRI, &X86InstOperands::ri(RCX, 0xFF));
        assert_eq!(bytes, vec![0x48, 0x81, 0xE1, 0xFF, 0x00, 0x00, 0x00]);
    }

    #[test]
    fn test_or_rdx_imm32() {
        // OR RDX, 1: REX.W + 81 + ModRM(11 001 010) + imm32
        let bytes = encode(X86Opcode::OrRI, &X86InstOperands::ri(RDX, 1));
        assert_eq!(bytes, vec![0x48, 0x81, 0xCA, 0x01, 0x00, 0x00, 0x00]);
    }

    #[test]
    fn test_xor_rbx_imm32() {
        // XOR RBX, 0xDEAD: REX.W + 81 + ModRM(11 110 011) + imm32
        let bytes = encode(X86Opcode::XorRI, &X86InstOperands::ri(RBX, 0xDEAD));
        assert_eq!(bytes, vec![0x48, 0x81, 0xF3, 0xAD, 0xDE, 0x00, 0x00]);
    }

    // -----------------------------------------------------------------------
    // CMP / TEST tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_cmp_rax_rcx() {
        // CMP RAX, RCX: REX.W + 39 + ModRM(11 001 000) = C8
        let bytes = encode(X86Opcode::CmpRR, &X86InstOperands::rr(RAX, RCX));
        assert_eq!(bytes, vec![0x48, 0x39, 0xC8]);
    }

    #[test]
    fn test_cmp_rax_imm32() {
        // CMP RAX, 0: REX.W + 81 + ModRM(11 111 000) + imm32
        let bytes = encode(X86Opcode::CmpRI, &X86InstOperands::ri(RAX, 0));
        assert_eq!(bytes, vec![0x48, 0x81, 0xF8, 0x00, 0x00, 0x00, 0x00]);
    }

    #[test]
    fn test_test_rax_rcx() {
        // TEST RAX, RCX: REX.W + 85 + ModRM(11 001 000)
        let bytes = encode(X86Opcode::TestRR, &X86InstOperands::rr(RAX, RCX));
        assert_eq!(bytes, vec![0x48, 0x85, 0xC8]);
    }

    #[test]
    fn test_test_rax_imm32() {
        // TEST RAX, 1: REX.W + F7 + ModRM(11 000 000) + imm32
        let bytes = encode(X86Opcode::TestRI, &X86InstOperands::ri(RAX, 1));
        assert_eq!(bytes, vec![0x48, 0xF7, 0xC0, 0x01, 0x00, 0x00, 0x00]);
    }

    // -----------------------------------------------------------------------
    // MOV tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_mov_rax_rcx() {
        // MOV RAX, RCX: REX.W + 89 + ModRM(11 001 000) = C8
        let bytes = encode(X86Opcode::MovRR, &X86InstOperands::rr(RAX, RCX));
        assert_eq!(bytes, vec![0x48, 0x89, 0xC8]);
    }

    #[test]
    fn test_mov_r15_r14() {
        // MOV R15, R14: REX.WRB(4D) + 89 + ModRM(11 110 111) = F7
        let bytes = encode(X86Opcode::MovRR, &X86InstOperands::rr(R15, R14));
        assert_eq!(bytes, vec![0x4D, 0x89, 0xF7]);
    }

    #[test]
    fn test_movabs_rax_imm64() {
        // MOV RAX, 0x123456789ABCDEF0: REX.W + B8 + imm64
        let bytes = encode(
            X86Opcode::MovRI,
            &X86InstOperands::ri(RAX, 0x123456789ABCDEF0u64 as i64),
        );
        assert_eq!(
            bytes,
            vec![0x48, 0xB8, 0xF0, 0xDE, 0xBC, 0x9A, 0x78, 0x56, 0x34, 0x12]
        );
    }

    #[test]
    fn test_movabs_r8_imm64() {
        // MOV R8, 42: REX.WB(49) + B8 + imm64
        let bytes = encode(X86Opcode::MovRI, &X86InstOperands::ri(R8, 42));
        assert_eq!(
            bytes,
            vec![0x49, 0xB8, 0x2A, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
        );
    }

    // -----------------------------------------------------------------------
    // MOV memory tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_mov_rax_mem_rbx() {
        // MOV RAX, [RBX]: REX.W + 8B + ModRM(00 000 011)
        let bytes = encode(X86Opcode::MovRM, &X86InstOperands::rm(RAX, RBX, 0));
        assert_eq!(bytes, vec![0x48, 0x8B, 0x03]);
    }

    #[test]
    fn test_mov_rax_mem_rbx_disp8() {
        // MOV RAX, [RBX+16]: REX.W + 8B + ModRM(01 000 011) + disp8(10)
        let bytes = encode(X86Opcode::MovRM, &X86InstOperands::rm(RAX, RBX, 16));
        assert_eq!(bytes, vec![0x48, 0x8B, 0x43, 0x10]);
    }

    #[test]
    fn test_mov_rax_mem_rbx_disp32() {
        // MOV RAX, [RBX+256]: REX.W + 8B + ModRM(10 000 011) + disp32
        let bytes = encode(X86Opcode::MovRM, &X86InstOperands::rm(RAX, RBX, 256));
        assert_eq!(bytes, vec![0x48, 0x8B, 0x83, 0x00, 0x01, 0x00, 0x00]);
    }

    #[test]
    fn test_mov_rax_mem_rsp() {
        // MOV RAX, [RSP]: REX.W + 8B + ModRM(00 000 100) + SIB(00 100 100)
        // RSP as base requires SIB byte
        let bytes = encode(X86Opcode::MovRM, &X86InstOperands::rm(RAX, RSP, 0));
        assert_eq!(bytes, vec![0x48, 0x8B, 0x04, 0x24]);
    }

    #[test]
    fn test_mov_rax_mem_rsp_disp8() {
        // MOV RAX, [RSP+8]: REX.W + 8B + ModRM(01 000 100) + SIB(00 100 100) + disp8(08)
        let bytes = encode(X86Opcode::MovRM, &X86InstOperands::rm(RAX, RSP, 8));
        assert_eq!(bytes, vec![0x48, 0x8B, 0x44, 0x24, 0x08]);
    }

    #[test]
    fn test_mov_rax_mem_rbp() {
        // MOV RAX, [RBP+0]: RBP as base with disp=0 requires disp8=0
        // REX.W + 8B + ModRM(01 000 101) + disp8(00)
        let bytes = encode(X86Opcode::MovRM, &X86InstOperands::rm(RAX, RBP, 0));
        assert_eq!(bytes, vec![0x48, 0x8B, 0x45, 0x00]);
    }

    #[test]
    fn test_mov_mem_rbx_rax() {
        // MOV [RBX], RAX: REX.W + 89 + ModRM(00 000 011)
        // For MovMR, dst field holds the source register
        let bytes = encode(
            X86Opcode::MovMR,
            &X86InstOperands {
                dst: Some(RAX),
                base: Some(RBX),
                disp: 0,
                ..X86InstOperands::none()
            },
        );
        assert_eq!(bytes, vec![0x48, 0x89, 0x03]);
    }

    // -----------------------------------------------------------------------
    // IMUL tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_imul_rax_rcx() {
        // IMUL RAX, RCX: REX.W + 0F AF + ModRM(11 000 001)
        let bytes = encode(X86Opcode::ImulRR, &X86InstOperands::rr(RAX, RCX));
        assert_eq!(bytes, vec![0x48, 0x0F, 0xAF, 0xC1]);
    }

    #[test]
    fn test_imul_r8_r9() {
        // IMUL R8, R9: REX.WRB(4D) + 0F AF + ModRM(11 000 001)
        let bytes = encode(X86Opcode::ImulRR, &X86InstOperands::rr(R8, R9));
        assert_eq!(bytes, vec![0x4D, 0x0F, 0xAF, 0xC1]);
    }

    #[test]
    fn test_imul_rax_rcx_imm32() {
        // IMUL RAX, RCX, 42: REX.W + 69 + ModRM(11 000 001) + imm32
        let bytes = encode(
            X86Opcode::ImulRRI,
            &X86InstOperands::rri(RAX, RCX, 42),
        );
        assert_eq!(bytes, vec![0x48, 0x69, 0xC1, 0x2A, 0x00, 0x00, 0x00]);
    }

    // -----------------------------------------------------------------------
    // Unary tests: NEG, NOT, INC, DEC, IDIV
    // -----------------------------------------------------------------------

    #[test]
    fn test_neg_rax() {
        // NEG RAX: REX.W + F7 + ModRM(11 011 000)
        let bytes = encode(X86Opcode::Neg, &X86InstOperands::r(RAX));
        assert_eq!(bytes, vec![0x48, 0xF7, 0xD8]);
    }

    #[test]
    fn test_neg_r15() {
        // NEG R15: REX.WB(49) + F7 + ModRM(11 011 111)
        let bytes = encode(X86Opcode::Neg, &X86InstOperands::r(R15));
        assert_eq!(bytes, vec![0x49, 0xF7, 0xDF]);
    }

    #[test]
    fn test_not_rcx() {
        // NOT RCX: REX.W + F7 + ModRM(11 010 001)
        let bytes = encode(X86Opcode::Not, &X86InstOperands::r(RCX));
        assert_eq!(bytes, vec![0x48, 0xF7, 0xD1]);
    }

    #[test]
    fn test_inc_rax() {
        // INC RAX: REX.W + FF + ModRM(11 000 000)
        let bytes = encode(X86Opcode::Inc, &X86InstOperands::r(RAX));
        assert_eq!(bytes, vec![0x48, 0xFF, 0xC0]);
    }

    #[test]
    fn test_dec_rcx() {
        // DEC RCX: REX.W + FF + ModRM(11 001 001)
        let bytes = encode(X86Opcode::Dec, &X86InstOperands::r(RCX));
        assert_eq!(bytes, vec![0x48, 0xFF, 0xC9]);
    }

    #[test]
    fn test_idiv_rcx() {
        // IDIV RCX: REX.W + F7 + ModRM(11 111 001)
        let bytes = encode(X86Opcode::Idiv, &X86InstOperands::r(RCX));
        assert_eq!(bytes, vec![0x48, 0xF7, 0xF9]);
    }

    #[test]
    fn test_div_rcx() {
        // DIV RCX: REX.W + F7 + ModRM(11 110 001)
        // ModR/M: mod=11, reg=/6(110), rm=RCX(001) = 0xF1
        let bytes = encode(X86Opcode::Div, &X86InstOperands::r(RCX));
        assert_eq!(bytes, vec![0x48, 0xF7, 0xF1]);
    }

    #[test]
    fn test_div_r8() {
        // DIV R8: REX.WB(49) + F7 + ModRM(11 110 000)
        // R8 hw_enc=8, bit3=1 -> REX.B. ModR/M rm=R8(0 low3)
        let bytes = encode(X86Opcode::Div, &X86InstOperands::r(R8));
        assert_eq!(bytes, vec![0x49, 0xF7, 0xF0]);
    }

    // -----------------------------------------------------------------------
    // Shift tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_shl_rax_imm() {
        // SHL RAX, 4: REX.W + C1 + ModRM(11 100 000) + ib(04)
        let bytes = encode(X86Opcode::ShlRI, &X86InstOperands::ri(RAX, 4));
        assert_eq!(bytes, vec![0x48, 0xC1, 0xE0, 0x04]);
    }

    #[test]
    fn test_shr_rdx_imm() {
        // SHR RDX, 8: REX.W + C1 + ModRM(11 101 010) + ib(08)
        let bytes = encode(X86Opcode::ShrRI, &X86InstOperands::ri(RDX, 8));
        assert_eq!(bytes, vec![0x48, 0xC1, 0xEA, 0x08]);
    }

    #[test]
    fn test_sar_rcx_imm() {
        // SAR RCX, 1: REX.W + C1 + ModRM(11 111 001) + ib(01)
        let bytes = encode(X86Opcode::SarRI, &X86InstOperands::ri(RCX, 1));
        assert_eq!(bytes, vec![0x48, 0xC1, 0xF9, 0x01]);
    }

    #[test]
    fn test_shl_rax_cl() {
        // SHL RAX, CL: REX.W + D3 + ModRM(11 100 000)
        let bytes = encode(X86Opcode::ShlRR, &X86InstOperands::r(RAX));
        assert_eq!(bytes, vec![0x48, 0xD3, 0xE0]);
    }

    #[test]
    fn test_shr_rdx_cl() {
        // SHR RDX, CL: REX.W + D3 + ModRM(11 101 010)
        let bytes = encode(X86Opcode::ShrRR, &X86InstOperands::r(RDX));
        assert_eq!(bytes, vec![0x48, 0xD3, 0xEA]);
    }

    #[test]
    fn test_sar_r15_cl() {
        // SAR R15, CL: REX.WB(49) + D3 + ModRM(11 111 111)
        let bytes = encode(X86Opcode::SarRR, &X86InstOperands::r(R15));
        assert_eq!(bytes, vec![0x49, 0xD3, 0xFF]);
    }

    // -----------------------------------------------------------------------
    // Control flow tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_ret() {
        let bytes = encode(X86Opcode::Ret, &X86InstOperands::none());
        assert_eq!(bytes, vec![0xC3]);
    }

    #[test]
    fn test_call_rel32() {
        // CALL +0: E8 00000000
        let bytes = encode(X86Opcode::Call, &X86InstOperands::rel(0));
        assert_eq!(bytes, vec![0xE8, 0x00, 0x00, 0x00, 0x00]);
    }

    #[test]
    fn test_call_rel32_offset() {
        // CALL +256: E8 00010000
        let bytes = encode(X86Opcode::Call, &X86InstOperands::rel(256));
        assert_eq!(bytes, vec![0xE8, 0x00, 0x01, 0x00, 0x00]);
    }

    #[test]
    fn test_call_r_rax() {
        // CALL RAX: FF + ModRM(11 010 000) = D0
        let bytes = encode(X86Opcode::CallR, &X86InstOperands::r(RAX));
        assert_eq!(bytes, vec![0xFF, 0xD0]);
    }

    #[test]
    fn test_call_r_r15() {
        // CALL R15: REX.B(41) + FF + ModRM(11 010 111) = D7
        let bytes = encode(X86Opcode::CallR, &X86InstOperands::r(R15));
        assert_eq!(bytes, vec![0x41, 0xFF, 0xD7]);
    }

    #[test]
    fn test_jmp_rel32() {
        // JMP +0: E9 00000000
        let bytes = encode(X86Opcode::Jmp, &X86InstOperands::rel(0));
        assert_eq!(bytes, vec![0xE9, 0x00, 0x00, 0x00, 0x00]);
    }

    #[test]
    fn test_jcc_je() {
        // JE +0: 0F 84 00000000
        let bytes = encode(
            X86Opcode::Jcc,
            &X86InstOperands::jcc(X86CondCode::E, 0),
        );
        assert_eq!(bytes, vec![0x0F, 0x84, 0x00, 0x00, 0x00, 0x00]);
    }

    #[test]
    fn test_jcc_jne() {
        // JNE +100: 0F 85 64000000
        let bytes = encode(
            X86Opcode::Jcc,
            &X86InstOperands::jcc(X86CondCode::NE, 100),
        );
        assert_eq!(bytes, vec![0x0F, 0x85, 0x64, 0x00, 0x00, 0x00]);
    }

    #[test]
    fn test_jcc_jl() {
        // JL -16: 0F 8C F0FFFFFF
        let bytes = encode(
            X86Opcode::Jcc,
            &X86InstOperands::jcc(X86CondCode::L, -16),
        );
        assert_eq!(bytes, vec![0x0F, 0x8C, 0xF0, 0xFF, 0xFF, 0xFF]);
    }

    #[test]
    fn test_jcc_jg() {
        // JG +0: 0F 8F 00000000
        let bytes = encode(
            X86Opcode::Jcc,
            &X86InstOperands::jcc(X86CondCode::G, 0),
        );
        assert_eq!(bytes, vec![0x0F, 0x8F, 0x00, 0x00, 0x00, 0x00]);
    }

    // -----------------------------------------------------------------------
    // Stack tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_push_rax() {
        // PUSH RAX: 50
        let bytes = encode(X86Opcode::Push, &X86InstOperands::r(RAX));
        assert_eq!(bytes, vec![0x50]);
    }

    #[test]
    fn test_push_rbx() {
        // PUSH RBX: 53
        let bytes = encode(X86Opcode::Push, &X86InstOperands::r(RBX));
        assert_eq!(bytes, vec![0x53]);
    }

    #[test]
    fn test_push_r8() {
        // PUSH R8: REX.B(41) + 50
        let bytes = encode(X86Opcode::Push, &X86InstOperands::r(R8));
        assert_eq!(bytes, vec![0x41, 0x50]);
    }

    #[test]
    fn test_push_r15() {
        // PUSH R15: REX.B(41) + 57
        let bytes = encode(X86Opcode::Push, &X86InstOperands::r(R15));
        assert_eq!(bytes, vec![0x41, 0x57]);
    }

    #[test]
    fn test_pop_rax() {
        // POP RAX: 58
        let bytes = encode(X86Opcode::Pop, &X86InstOperands::r(RAX));
        assert_eq!(bytes, vec![0x58]);
    }

    #[test]
    fn test_pop_r15() {
        // POP R15: REX.B(41) + 5F
        let bytes = encode(X86Opcode::Pop, &X86InstOperands::r(R15));
        assert_eq!(bytes, vec![0x41, 0x5F]);
    }

    // -----------------------------------------------------------------------
    // Extended register encoding correctness
    // -----------------------------------------------------------------------

    #[test]
    fn test_mov_r13_r14() {
        // MOV R13, R14: REX.WRB(4D) + 89 + ModRM(11 110 101)
        let bytes = encode(X86Opcode::MovRR, &X86InstOperands::rr(R13, R14));
        assert_eq!(bytes, vec![0x4D, 0x89, 0xF5]);
    }

    #[test]
    fn test_xor_r10_r11() {
        // XOR R10, R11: REX.WRB(4D) + 31 + ModRM(11 011 010)
        let bytes = encode(X86Opcode::XorRR, &X86InstOperands::rr(R10, R11));
        assert_eq!(bytes, vec![0x4D, 0x31, 0xDA]);
    }

    #[test]
    fn test_sub_rsi_rdi() {
        // SUB RSI, RDI: REX.W(48) + 29 + ModRM(11 111 110) = FE
        let bytes = encode(X86Opcode::SubRR, &X86InstOperands::rr(RSI, RDI));
        assert_eq!(bytes, vec![0x48, 0x29, 0xFE]);
    }

    // -----------------------------------------------------------------------
    // Error handling tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_missing_dst_error() {
        let mut enc = X86Encoder::new();
        let result = enc.encode_instruction(X86Opcode::AddRR, &X86InstOperands::none());
        assert!(result.is_err());
    }

    #[test]
    fn test_missing_src_error() {
        let mut enc = X86Encoder::new();
        let result = enc.encode_instruction(X86Opcode::AddRR, &X86InstOperands::r(RAX));
        assert!(result.is_err());
    }

    #[test]
    fn test_missing_cc_error() {
        let mut enc = X86Encoder::new();
        let result = enc.encode_instruction(X86Opcode::Jcc, &X86InstOperands::rel(0));
        assert!(result.is_err());
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

    // -----------------------------------------------------------------------
    // Memory encoding with extended registers
    // -----------------------------------------------------------------------

    #[test]
    fn test_mov_r8_mem_r12() {
        // MOV R8, [R12]: REX.WRB(4D) + 8B + ModRM(00 000 100) + SIB(00 100 100)
        // R12 base uses SIB (hw_enc & 7 == 4)
        let bytes = encode(X86Opcode::MovRM, &X86InstOperands::rm(R8, R12, 0));
        assert_eq!(bytes, vec![0x4D, 0x8B, 0x04, 0x24]);
    }

    #[test]
    fn test_mov_r8_mem_r13() {
        // MOV R8, [R13+0]: R13 base with disp=0 requires disp8=0 (hw_enc & 7 == 5)
        // REX.WRB(4D) + 8B + ModRM(01 000 101) + disp8(00)
        let bytes = encode(X86Opcode::MovRM, &X86InstOperands::rm(R8, R13, 0));
        assert_eq!(bytes, vec![0x4D, 0x8B, 0x45, 0x00]);
    }

    // -----------------------------------------------------------------------
    // AddRM, SubRM, CmpRM tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_add_rax_mem_rbx() {
        // ADD RAX, [RBX]: REX.W + 03 + ModRM(00 000 011)
        let bytes = encode(X86Opcode::AddRM, &X86InstOperands::rm(RAX, RBX, 0));
        assert_eq!(bytes, vec![0x48, 0x03, 0x03]);
    }

    #[test]
    fn test_sub_rcx_mem_rdx_disp() {
        // SUB RCX, [RDX+16]: REX.W + 2B + ModRM(01 001 010) + disp8(10)
        let bytes = encode(X86Opcode::SubRM, &X86InstOperands::rm(RCX, RDX, 16));
        assert_eq!(bytes, vec![0x48, 0x2B, 0x4A, 0x10]);
    }

    #[test]
    fn test_cmp_rdi_mem_rsi() {
        // CMP RDI, [RSI]: REX.W + 3B + ModRM(00 111 110)
        let bytes = encode(X86Opcode::CmpRM, &X86InstOperands::rm(RDI, RSI, 0));
        assert_eq!(bytes, vec![0x48, 0x3B, 0x3E]);
    }

    // -----------------------------------------------------------------------
    // Instruction size tests (verify correct byte counts)
    // -----------------------------------------------------------------------

    #[test]
    fn test_instruction_sizes() {
        let mut enc = X86Encoder::new();

        // RET = 1 byte
        let n = enc.encode_instruction(X86Opcode::Ret, &X86InstOperands::none()).unwrap();
        assert_eq!(n, 1);

        // PUSH RAX = 1 byte (no REX needed)
        let n = enc.encode_instruction(X86Opcode::Push, &X86InstOperands::r(RAX)).unwrap();
        assert_eq!(n, 1);

        // PUSH R8 = 2 bytes (REX.B + opcode)
        let n = enc.encode_instruction(X86Opcode::Push, &X86InstOperands::r(R8)).unwrap();
        assert_eq!(n, 2);

        // ADD RAX, RCX = 3 bytes (REX.W + opcode + ModRM)
        let n = enc.encode_instruction(X86Opcode::AddRR, &X86InstOperands::rr(RAX, RCX)).unwrap();
        assert_eq!(n, 3);

        // CALL rel32 = 5 bytes (opcode + imm32)
        let n = enc.encode_instruction(X86Opcode::Call, &X86InstOperands::rel(0)).unwrap();
        assert_eq!(n, 5);

        // Jcc rel32 = 6 bytes (0F + opcode + imm32)
        let n = enc.encode_instruction(
            X86Opcode::Jcc,
            &X86InstOperands::jcc(X86CondCode::E, 0),
        ).unwrap();
        assert_eq!(n, 6);

        // ADD RAX, imm32 = 7 bytes (REX.W + opcode + ModRM + imm32)
        let n = enc.encode_instruction(X86Opcode::AddRI, &X86InstOperands::ri(RAX, 42)).unwrap();
        assert_eq!(n, 7);

        // MOV RAX, imm64 = 10 bytes (REX.W + opcode + imm64)
        let n = enc.encode_instruction(X86Opcode::MovRI, &X86InstOperands::ri(RAX, 42)).unwrap();
        assert_eq!(n, 10);
    }

    // -----------------------------------------------------------------------
    // LEA tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_lea_rax_rbx() {
        // LEA RAX, [RBX]: REX.W(48) + 8D + ModRM(00 000 011)
        let bytes = encode(X86Opcode::Lea, &X86InstOperands::rm(RAX, RBX, 0));
        assert_eq!(bytes, vec![0x48, 0x8D, 0x03]);
    }

    #[test]
    fn test_lea_rax_rbx_disp8() {
        // LEA RAX, [RBX+16]: REX.W + 8D + ModRM(01 000 011) + disp8
        let bytes = encode(X86Opcode::Lea, &X86InstOperands::rm(RAX, RBX, 16));
        assert_eq!(bytes, vec![0x48, 0x8D, 0x43, 0x10]);
    }

    #[test]
    fn test_lea_rax_rbx_disp32() {
        // LEA RAX, [RBX+256]: REX.W + 8D + ModRM(10 000 011) + disp32
        let bytes = encode(X86Opcode::Lea, &X86InstOperands::rm(RAX, RBX, 256));
        assert_eq!(bytes, vec![0x48, 0x8D, 0x83, 0x00, 0x01, 0x00, 0x00]);
    }

    #[test]
    fn test_lea_r15_rsp_disp8() {
        // LEA R15, [RSP+8]: REX.WRB(4C) + 8D + ModRM(01 111 100) + SIB(00 100 100) + disp8
        let bytes = encode(X86Opcode::Lea, &X86InstOperands::rm(R15, RSP, 8));
        assert_eq!(bytes, vec![0x4C, 0x8D, 0x7C, 0x24, 0x08]);
    }

    // -----------------------------------------------------------------------
    // MOVZX / MOVSX tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_movzx_rax_cl() {
        // MOVZX RAX, CL: REX.W(48) + 0F B6 + ModRM(11 000 001)
        let bytes = encode(X86Opcode::Movzx, &X86InstOperands::rr(RAX, RCX));
        assert_eq!(bytes, vec![0x48, 0x0F, 0xB6, 0xC1]);
    }

    #[test]
    fn test_movzx_r8_al() {
        // MOVZX R8, AL: REX.WR(4C) + 0F B6 + ModRM(11 000 000)
        let bytes = encode(X86Opcode::Movzx, &X86InstOperands::rr(R8, RAX));
        assert_eq!(bytes, vec![0x4C, 0x0F, 0xB6, 0xC0]);
    }

    #[test]
    fn test_movsx_rax_ecx() {
        // MOVSXD RAX, ECX: REX.W(48) + 63 + ModRM(11 000 001)
        let bytes = encode(X86Opcode::Movsx, &X86InstOperands::rr(RAX, RCX));
        assert_eq!(bytes, vec![0x48, 0x63, 0xC1]);
    }

    #[test]
    fn test_movsx_r15_r14() {
        // MOVSXD R15, R14: REX.WRB(4D) + 63 + ModRM(11 111 110)
        let bytes = encode(X86Opcode::Movsx, &X86InstOperands::rr(R15, R14));
        assert_eq!(bytes, vec![0x4D, 0x63, 0xFE]);
    }

    // -----------------------------------------------------------------------
    // SSE scalar double-precision tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_addsd_xmm0_xmm1() {
        // ADDSD XMM0, XMM1: F2 0F 58 + ModRM(11 000 001)
        let bytes = encode(X86Opcode::Addsd, &X86InstOperands::rr(XMM0, XMM1));
        assert_eq!(bytes, vec![0xF2, 0x0F, 0x58, 0xC1]);
    }

    #[test]
    fn test_subsd_xmm0_xmm1() {
        // SUBSD XMM0, XMM1: F2 0F 5C + ModRM(11 000 001)
        let bytes = encode(X86Opcode::Subsd, &X86InstOperands::rr(XMM0, XMM1));
        assert_eq!(bytes, vec![0xF2, 0x0F, 0x5C, 0xC1]);
    }

    #[test]
    fn test_mulsd_xmm0_xmm1() {
        // MULSD XMM0, XMM1: F2 0F 59 + ModRM(11 000 001)
        let bytes = encode(X86Opcode::Mulsd, &X86InstOperands::rr(XMM0, XMM1));
        assert_eq!(bytes, vec![0xF2, 0x0F, 0x59, 0xC1]);
    }

    #[test]
    fn test_divsd_xmm0_xmm1() {
        // DIVSD XMM0, XMM1: F2 0F 5E + ModRM(11 000 001)
        let bytes = encode(X86Opcode::Divsd, &X86InstOperands::rr(XMM0, XMM1));
        assert_eq!(bytes, vec![0xF2, 0x0F, 0x5E, 0xC1]);
    }

    #[test]
    fn test_addsd_xmm8_xmm15() {
        // ADDSD XMM8, XMM15: F2 REX.RB(45) 0F 58 + ModRM(11 000 111)
        let bytes = encode(X86Opcode::Addsd, &X86InstOperands::rr(XMM8, XMM15));
        assert_eq!(bytes, vec![0xF2, 0x45, 0x0F, 0x58, 0xC7]);
    }

    #[test]
    fn test_movsd_xmm0_xmm1() {
        // MOVSD XMM0, XMM1: F2 0F 10 + ModRM(11 000 001)
        let bytes = encode(X86Opcode::MovsdRR, &X86InstOperands::rr(XMM0, XMM1));
        assert_eq!(bytes, vec![0xF2, 0x0F, 0x10, 0xC1]);
    }

    #[test]
    fn test_movsd_rm_xmm0_rbx() {
        // MOVSD XMM0, [RBX]: F2 0F 10 + ModRM(00 000 011)
        let bytes = encode(X86Opcode::MovsdRM, &X86InstOperands::rm(XMM0, RBX, 0));
        assert_eq!(bytes, vec![0xF2, 0x0F, 0x10, 0x03]);
    }

    #[test]
    fn test_movsd_rm_xmm0_rbx_disp8() {
        // MOVSD XMM0, [RBX+8]: F2 0F 10 + ModRM(01 000 011) + disp8
        let bytes = encode(X86Opcode::MovsdRM, &X86InstOperands::rm(XMM0, RBX, 8));
        assert_eq!(bytes, vec![0xF2, 0x0F, 0x10, 0x43, 0x08]);
    }

    #[test]
    fn test_movsd_mr_rbx_xmm0() {
        // MOVSD [RBX], XMM0: F2 0F 11 + ModRM(00 000 011)
        // For MovsdMR, dst field holds the source XMM register.
        let bytes = encode(
            X86Opcode::MovsdMR,
            &X86InstOperands {
                dst: Some(XMM0),
                base: Some(RBX),
                disp: 0,
                ..X86InstOperands::none()
            },
        );
        assert_eq!(bytes, vec![0xF2, 0x0F, 0x11, 0x03]);
    }

    #[test]
    fn test_ucomisd_xmm0_xmm1() {
        // UCOMISD XMM0, XMM1: 66 0F 2E + ModRM(11 000 001)
        let bytes = encode(X86Opcode::Ucomisd, &X86InstOperands::rr(XMM0, XMM1));
        assert_eq!(bytes, vec![0x66, 0x0F, 0x2E, 0xC1]);
    }

    // -----------------------------------------------------------------------
    // SSE scalar single-precision tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_addss_xmm0_xmm1() {
        // ADDSS XMM0, XMM1: F3 0F 58 + ModRM(11 000 001)
        let bytes = encode(X86Opcode::Addss, &X86InstOperands::rr(XMM0, XMM1));
        assert_eq!(bytes, vec![0xF3, 0x0F, 0x58, 0xC1]);
    }

    #[test]
    fn test_subss_xmm0_xmm1() {
        // SUBSS XMM0, XMM1: F3 0F 5C + ModRM(11 000 001)
        let bytes = encode(X86Opcode::Subss, &X86InstOperands::rr(XMM0, XMM1));
        assert_eq!(bytes, vec![0xF3, 0x0F, 0x5C, 0xC1]);
    }

    #[test]
    fn test_mulss_xmm0_xmm1() {
        // MULSS XMM0, XMM1: F3 0F 59 + ModRM(11 000 001)
        let bytes = encode(X86Opcode::Mulss, &X86InstOperands::rr(XMM0, XMM1));
        assert_eq!(bytes, vec![0xF3, 0x0F, 0x59, 0xC1]);
    }

    #[test]
    fn test_divss_xmm0_xmm1() {
        // DIVSS XMM0, XMM1: F3 0F 5E + ModRM(11 000 001)
        let bytes = encode(X86Opcode::Divss, &X86InstOperands::rr(XMM0, XMM1));
        assert_eq!(bytes, vec![0xF3, 0x0F, 0x5E, 0xC1]);
    }

    #[test]
    fn test_addss_xmm8_xmm15() {
        // ADDSS XMM8, XMM15: F3 REX.RB(45) 0F 58 + ModRM(11 000 111)
        let bytes = encode(X86Opcode::Addss, &X86InstOperands::rr(XMM8, XMM15));
        assert_eq!(bytes, vec![0xF3, 0x45, 0x0F, 0x58, 0xC7]);
    }

    #[test]
    fn test_movss_xmm0_xmm1() {
        // MOVSS XMM0, XMM1: F3 0F 10 + ModRM(11 000 001)
        let bytes = encode(X86Opcode::MovssRR, &X86InstOperands::rr(XMM0, XMM1));
        assert_eq!(bytes, vec![0xF3, 0x0F, 0x10, 0xC1]);
    }

    #[test]
    fn test_movss_rm_xmm0_rbx() {
        // MOVSS XMM0, [RBX]: F3 0F 10 + ModRM(00 000 011)
        let bytes = encode(X86Opcode::MovssRM, &X86InstOperands::rm(XMM0, RBX, 0));
        assert_eq!(bytes, vec![0xF3, 0x0F, 0x10, 0x03]);
    }

    #[test]
    fn test_movss_mr_rbx_xmm0() {
        // MOVSS [RBX], XMM0: F3 0F 11 + ModRM(00 000 011)
        let bytes = encode(
            X86Opcode::MovssMR,
            &X86InstOperands {
                dst: Some(XMM0),
                base: Some(RBX),
                disp: 0,
                ..X86InstOperands::none()
            },
        );
        assert_eq!(bytes, vec![0xF3, 0x0F, 0x11, 0x03]);
    }

    #[test]
    fn test_ucomiss_xmm0_xmm1() {
        // UCOMISS XMM0, XMM1: 0F 2E + ModRM(11 000 001) (no prefix)
        let bytes = encode(X86Opcode::Ucomiss, &X86InstOperands::rr(XMM0, XMM1));
        assert_eq!(bytes, vec![0x0F, 0x2E, 0xC1]);
    }

    #[test]
    fn test_ucomiss_xmm8_xmm15() {
        // UCOMISS XMM8, XMM15: REX.RB(45) 0F 2E + ModRM(11 000 111)
        let bytes = encode(X86Opcode::Ucomiss, &X86InstOperands::rr(XMM8, XMM15));
        assert_eq!(bytes, vec![0x45, 0x0F, 0x2E, 0xC7]);
    }

    // -----------------------------------------------------------------------
    // CMOVcc tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_cmove_rax_rcx() {
        // CMOVE RAX, RCX: REX.W(48) + 0F 44 + ModRM(11 000 001)
        let bytes = encode(
            X86Opcode::Cmovcc,
            &X86InstOperands {
                dst: Some(RAX),
                src: Some(RCX),
                cc: Some(X86CondCode::E),
                ..X86InstOperands::none()
            },
        );
        assert_eq!(bytes, vec![0x48, 0x0F, 0x44, 0xC1]);
    }

    #[test]
    fn test_cmovne_rbx_rdx() {
        // CMOVNE RBX, RDX: REX.W(48) + 0F 45 + ModRM(11 011 010)
        let bytes = encode(
            X86Opcode::Cmovcc,
            &X86InstOperands {
                dst: Some(RBX),
                src: Some(RDX),
                cc: Some(X86CondCode::NE),
                ..X86InstOperands::none()
            },
        );
        assert_eq!(bytes, vec![0x48, 0x0F, 0x45, 0xDA]);
    }

    #[test]
    fn test_cmovl_r8_r9() {
        // CMOVL R8, R9: REX.WRB(4D) + 0F 4C + ModRM(11 000 001)
        let bytes = encode(
            X86Opcode::Cmovcc,
            &X86InstOperands {
                dst: Some(R8),
                src: Some(R9),
                cc: Some(X86CondCode::L),
                ..X86InstOperands::none()
            },
        );
        assert_eq!(bytes, vec![0x4D, 0x0F, 0x4C, 0xC1]);
    }

    #[test]
    fn test_cmovg_rax_r15() {
        // CMOVG RAX, R15: REX.WB(49) + 0F 4F + ModRM(11 000 111)
        let bytes = encode(
            X86Opcode::Cmovcc,
            &X86InstOperands {
                dst: Some(RAX),
                src: Some(R15),
                cc: Some(X86CondCode::G),
                ..X86InstOperands::none()
            },
        );
        assert_eq!(bytes, vec![0x49, 0x0F, 0x4F, 0xC7]);
    }

    // -----------------------------------------------------------------------
    // SETcc tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_sete_al() {
        // SETE AL: 0F 94 + ModRM(11 000 000)
        let bytes = encode(
            X86Opcode::Setcc,
            &X86InstOperands {
                dst: Some(AL),
                cc: Some(X86CondCode::E),
                ..X86InstOperands::none()
            },
        );
        assert_eq!(bytes, vec![0x0F, 0x94, 0xC0]);
    }

    #[test]
    fn test_setne_cl() {
        // SETNE CL: 0F 95 + ModRM(11 000 001)
        let bytes = encode(
            X86Opcode::Setcc,
            &X86InstOperands {
                dst: Some(CL),
                cc: Some(X86CondCode::NE),
                ..X86InstOperands::none()
            },
        );
        assert_eq!(bytes, vec![0x0F, 0x95, 0xC1]);
    }

    #[test]
    fn test_setl_r8b() {
        // SETL R8B: REX.B(41) + 0F 9C + ModRM(11 000 000)
        let bytes = encode(
            X86Opcode::Setcc,
            &X86InstOperands {
                dst: Some(R8B),
                cc: Some(X86CondCode::L),
                ..X86InstOperands::none()
            },
        );
        assert_eq!(bytes, vec![0x41, 0x0F, 0x9C, 0xC0]);
    }

    #[test]
    fn test_setg_al() {
        // SETG AL: 0F 9F + ModRM(11 000 000)
        let bytes = encode(
            X86Opcode::Setcc,
            &X86InstOperands {
                dst: Some(AL),
                cc: Some(X86CondCode::G),
                ..X86InstOperands::none()
            },
        );
        assert_eq!(bytes, vec![0x0F, 0x9F, 0xC0]);
    }

    // -----------------------------------------------------------------------
    // Bit manipulation tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_bsf_rax_rcx() {
        // BSF RAX, RCX: REX.W(48) + 0F BC + ModRM(11 000 001)
        let bytes = encode(X86Opcode::Bsf, &X86InstOperands::rr(RAX, RCX));
        assert_eq!(bytes, vec![0x48, 0x0F, 0xBC, 0xC1]);
    }

    #[test]
    fn test_bsf_r8_r9() {
        // BSF R8, R9: REX.WRB(4D) + 0F BC + ModRM(11 000 001)
        let bytes = encode(X86Opcode::Bsf, &X86InstOperands::rr(R8, R9));
        assert_eq!(bytes, vec![0x4D, 0x0F, 0xBC, 0xC1]);
    }

    #[test]
    fn test_bsr_rax_rcx() {
        // BSR RAX, RCX: REX.W(48) + 0F BD + ModRM(11 000 001)
        let bytes = encode(X86Opcode::Bsr, &X86InstOperands::rr(RAX, RCX));
        assert_eq!(bytes, vec![0x48, 0x0F, 0xBD, 0xC1]);
    }

    #[test]
    fn test_bsr_r15_rax() {
        // BSR R15, RAX: REX.WR(4C) + 0F BD + ModRM(11 111 000)
        let bytes = encode(X86Opcode::Bsr, &X86InstOperands::rr(R15, RAX));
        assert_eq!(bytes, vec![0x4C, 0x0F, 0xBD, 0xF8]);
    }

    #[test]
    fn test_tzcnt_rax_rcx() {
        // TZCNT RAX, RCX: F3 REX.W(48) + 0F BC + ModRM(11 000 001)
        let bytes = encode(X86Opcode::Tzcnt, &X86InstOperands::rr(RAX, RCX));
        assert_eq!(bytes, vec![0xF3, 0x48, 0x0F, 0xBC, 0xC1]);
    }

    #[test]
    fn test_tzcnt_r8_r9() {
        // TZCNT R8, R9: F3 REX.WRB(4D) + 0F BC + ModRM(11 000 001)
        let bytes = encode(X86Opcode::Tzcnt, &X86InstOperands::rr(R8, R9));
        assert_eq!(bytes, vec![0xF3, 0x4D, 0x0F, 0xBC, 0xC1]);
    }

    #[test]
    fn test_lzcnt_rax_rcx() {
        // LZCNT RAX, RCX: F3 REX.W(48) + 0F BD + ModRM(11 000 001)
        let bytes = encode(X86Opcode::Lzcnt, &X86InstOperands::rr(RAX, RCX));
        assert_eq!(bytes, vec![0xF3, 0x48, 0x0F, 0xBD, 0xC1]);
    }

    #[test]
    fn test_lzcnt_r15_rax() {
        // LZCNT R15, RAX: F3 REX.WR(4C) + 0F BD + ModRM(11 111 000)
        let bytes = encode(X86Opcode::Lzcnt, &X86InstOperands::rr(R15, RAX));
        assert_eq!(bytes, vec![0xF3, 0x4C, 0x0F, 0xBD, 0xF8]);
    }

    #[test]
    fn test_popcnt_rax_rcx() {
        // POPCNT RAX, RCX: F3 REX.W(48) + 0F B8 + ModRM(11 000 001)
        let bytes = encode(X86Opcode::Popcnt, &X86InstOperands::rr(RAX, RCX));
        assert_eq!(bytes, vec![0xF3, 0x48, 0x0F, 0xB8, 0xC1]);
    }

    #[test]
    fn test_popcnt_r8_r9() {
        // POPCNT R8, R9: F3 REX.WRB(4D) + 0F B8 + ModRM(11 000 001)
        let bytes = encode(X86Opcode::Popcnt, &X86InstOperands::rr(R8, R9));
        assert_eq!(bytes, vec![0xF3, 0x4D, 0x0F, 0xB8, 0xC1]);
    }

    // -----------------------------------------------------------------------
    // Mixed encoding size tests for new instructions
    // -----------------------------------------------------------------------

    #[test]
    fn test_new_instruction_sizes() {
        let mut enc = X86Encoder::new();

        // SSE scalar: prefix(1) + 0F(1) + opcode(1) + ModRM(1) = 4 bytes
        let n = enc.encode_instruction(X86Opcode::Addsd, &X86InstOperands::rr(XMM0, XMM1)).unwrap();
        assert_eq!(n, 4);

        // SSE scalar with extended regs: prefix(1) + REX(1) + 0F(1) + opcode(1) + ModRM(1) = 5 bytes
        let n = enc.encode_instruction(X86Opcode::Addsd, &X86InstOperands::rr(XMM8, XMM15)).unwrap();
        assert_eq!(n, 5);

        // CMOVcc: REX.W(1) + 0F(1) + opcode(1) + ModRM(1) = 4 bytes
        let n = enc.encode_instruction(
            X86Opcode::Cmovcc,
            &X86InstOperands {
                dst: Some(RAX),
                src: Some(RCX),
                cc: Some(X86CondCode::E),
                ..X86InstOperands::none()
            },
        ).unwrap();
        assert_eq!(n, 4);

        // SETcc (no REX): 0F(1) + opcode(1) + ModRM(1) = 3 bytes
        let n = enc.encode_instruction(
            X86Opcode::Setcc,
            &X86InstOperands {
                dst: Some(AL),
                cc: Some(X86CondCode::E),
                ..X86InstOperands::none()
            },
        ).unwrap();
        assert_eq!(n, 3);

        // SETcc with REX.B: REX(1) + 0F(1) + opcode(1) + ModRM(1) = 4 bytes
        let n = enc.encode_instruction(
            X86Opcode::Setcc,
            &X86InstOperands {
                dst: Some(R8B),
                cc: Some(X86CondCode::E),
                ..X86InstOperands::none()
            },
        ).unwrap();
        assert_eq!(n, 4);

        // BSF: REX.W(1) + 0F(1) + opcode(1) + ModRM(1) = 4 bytes
        let n = enc.encode_instruction(X86Opcode::Bsf, &X86InstOperands::rr(RAX, RCX)).unwrap();
        assert_eq!(n, 4);

        // TZCNT: F3(1) + REX.W(1) + 0F(1) + opcode(1) + ModRM(1) = 5 bytes
        let n = enc.encode_instruction(X86Opcode::Tzcnt, &X86InstOperands::rr(RAX, RCX)).unwrap();
        assert_eq!(n, 5);

        // POPCNT: F3(1) + REX.W(1) + 0F(1) + opcode(1) + ModRM(1) = 5 bytes
        let n = enc.encode_instruction(X86Opcode::Popcnt, &X86InstOperands::rr(RAX, RCX)).unwrap();
        assert_eq!(n, 5);

        // LEA [base+0] = 3 bytes (REX.W + 8D + ModRM)
        let n = enc.encode_instruction(X86Opcode::Lea, &X86InstOperands::rm(RAX, RBX, 0)).unwrap();
        assert_eq!(n, 3);

        // MOVZX = 4 bytes (REX.W + 0F + B6 + ModRM)
        let n = enc.encode_instruction(X86Opcode::Movzx, &X86InstOperands::rr(RAX, RCX)).unwrap();
        assert_eq!(n, 4);

        // MOVSXD = 3 bytes (REX.W + 63 + ModRM)
        let n = enc.encode_instruction(X86Opcode::Movsx, &X86InstOperands::rr(RAX, RCX)).unwrap();
        assert_eq!(n, 3);
    }

    // -----------------------------------------------------------------------
    // CMOVcc / SETcc missing cc errors
    // -----------------------------------------------------------------------

    #[test]
    fn test_cmovcc_missing_cc_error() {
        let mut enc = X86Encoder::new();
        let result = enc.encode_instruction(
            X86Opcode::Cmovcc,
            &X86InstOperands::rr(RAX, RCX),
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_setcc_missing_cc_error() {
        let mut enc = X86Encoder::new();
        let result = enc.encode_instruction(
            X86Opcode::Setcc,
            &X86InstOperands::r(AL),
        );
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // All CMOVcc condition codes
    // -----------------------------------------------------------------------

    #[test]
    fn test_cmovcc_all_conditions() {
        let all_cc = [
            (X86CondCode::O,  0x40u8),
            (X86CondCode::NO, 0x41),
            (X86CondCode::B,  0x42),
            (X86CondCode::AE, 0x43),
            (X86CondCode::E,  0x44),
            (X86CondCode::NE, 0x45),
            (X86CondCode::BE, 0x46),
            (X86CondCode::A,  0x47),
            (X86CondCode::S,  0x48),
            (X86CondCode::NS, 0x49),
            (X86CondCode::P,  0x4A),
            (X86CondCode::NP, 0x4B),
            (X86CondCode::L,  0x4C),
            (X86CondCode::GE, 0x4D),
            (X86CondCode::LE, 0x4E),
            (X86CondCode::G,  0x4F),
        ];
        for (cc, expected_byte) in &all_cc {
            let bytes = encode(
                X86Opcode::Cmovcc,
                &X86InstOperands {
                    dst: Some(RAX),
                    src: Some(RCX),
                    cc: Some(*cc),
                    ..X86InstOperands::none()
                },
            );
            // REX.W(48) + 0F + cc_byte + ModRM
            assert_eq!(bytes[2], *expected_byte, "CMOVcc {:?}", cc);
        }
    }

    // -----------------------------------------------------------------------
    // SSE memory with extended registers
    // -----------------------------------------------------------------------

    #[test]
    fn test_movsd_rm_xmm8_rsp_disp8() {
        // MOVSD XMM8, [RSP+16]: F2 REX.R(44) 0F 10 + ModRM(01 000 100) + SIB(00 100 100) + disp8
        let bytes = encode(X86Opcode::MovsdRM, &X86InstOperands::rm(XMM8, RSP, 16));
        assert_eq!(bytes, vec![0xF2, 0x44, 0x0F, 0x10, 0x44, 0x24, 0x10]);
    }

    #[test]
    fn test_movss_rm_xmm8_rbp() {
        // MOVSS XMM8, [RBP+0]: F3 REX.R(44) 0F 10 + ModRM(01 000 101) + disp8(00)
        let bytes = encode(X86Opcode::MovssRM, &X86InstOperands::rm(XMM8, RBP, 0));
        assert_eq!(bytes, vec![0xF3, 0x44, 0x0F, 0x10, 0x45, 0x00]);
    }

    // -----------------------------------------------------------------------
    // RIP-relative LEA tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_lea_rip_rax_disp0() {
        // LEA RAX, [RIP+0]: REX.W(48) + 8D + ModRM(00 000 101) + disp32(00000000)
        let bytes = encode(X86Opcode::LeaRip, &X86InstOperands::rip_rel(RAX, 0));
        assert_eq!(bytes, vec![0x48, 0x8D, 0x05, 0x00, 0x00, 0x00, 0x00]);
    }

    #[test]
    fn test_lea_rip_rcx_disp256() {
        // LEA RCX, [RIP+256]: REX.W(48) + 8D + ModRM(00 001 101) + disp32
        let bytes = encode(X86Opcode::LeaRip, &X86InstOperands::rip_rel(RCX, 256));
        assert_eq!(bytes, vec![0x48, 0x8D, 0x0D, 0x00, 0x01, 0x00, 0x00]);
    }

    #[test]
    fn test_lea_rip_r8_negative() {
        // LEA R8, [RIP-16]: REX.WR(4C) + 8D + ModRM(00 000 101) + disp32(F0FFFFFF)
        let bytes = encode(X86Opcode::LeaRip, &X86InstOperands::rip_rel(R8, -16));
        assert_eq!(bytes, vec![0x4C, 0x8D, 0x05, 0xF0, 0xFF, 0xFF, 0xFF]);
    }

    // -----------------------------------------------------------------------
    // Scaled-index (SIB) memory tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_sib_scaled_encode() {
        // SIB for [RAX + RCX*4]: scale=2(4x), index=RCX(1), base=RAX(0)
        let sib = Sib::scaled(0, 1, 4);
        // scale=2(0b10), index=1(0b001), base=0(0b000) -> 10_001_000 = 0x88
        assert_eq!(sib.encode(), 0x88);
    }

    #[test]
    fn test_sib_scaled_encode_scale8() {
        // SIB for [RBX + RDX*8]: scale=3(8x), index=RDX(2), base=RBX(3)
        let sib = Sib::scaled(3, 2, 8);
        // scale=3(0b11), index=2(0b010), base=3(0b011) -> 11_010_011 = 0xD3
        assert_eq!(sib.encode(), 0xD3);
    }

    #[test]
    fn test_mov_rm_sib_rax_rbx_rcx_scale4_nodisp() {
        // MOV RAX, [RBX + RCX*4]: REX.W(48) + 8B + ModRM(00 000 100) + SIB(10 001 011)
        // reg=RAX(0), rm=100(SIB), SIB: scale=4(2), index=RCX(1), base=RBX(3)
        let bytes = encode(
            X86Opcode::MovRMSib,
            &X86InstOperands::rm_sib(RAX, RBX, RCX, 4, 0),
        );
        assert_eq!(bytes, vec![0x48, 0x8B, 0x04, 0x8B]);
    }

    #[test]
    fn test_mov_rm_sib_rax_rbx_rcx_scale4_disp8() {
        // MOV RAX, [RBX + RCX*4 + 16]: REX.W(48) + 8B + ModRM(01 000 100) + SIB + disp8
        let bytes = encode(
            X86Opcode::MovRMSib,
            &X86InstOperands::rm_sib(RAX, RBX, RCX, 4, 16),
        );
        assert_eq!(bytes, vec![0x48, 0x8B, 0x44, 0x8B, 0x10]);
    }

    #[test]
    fn test_mov_rm_sib_rax_rbx_rcx_scale8_disp32() {
        // MOV RAX, [RBX + RCX*8 + 256]: REX.W(48) + 8B + ModRM(10 000 100) + SIB + disp32
        let bytes = encode(
            X86Opcode::MovRMSib,
            &X86InstOperands::rm_sib(RAX, RBX, RCX, 8, 256),
        );
        assert_eq!(bytes, vec![0x48, 0x8B, 0x84, 0xCB, 0x00, 0x01, 0x00, 0x00]);
    }

    #[test]
    fn test_mov_rm_sib_r8_r12_r9_scale2() {
        // MOV R8, [R12 + R9*2]: REX.WRX.B(4F) + 8B + ModRM(00 000 100) + SIB
        // dst=R8(8, REX.R), base=R12(12, REX.B), index=R9(9, REX.X)
        let bytes = encode(
            X86Opcode::MovRMSib,
            &X86InstOperands::rm_sib(R8, R12, R9, 2, 0),
        );
        // REX: W=1, R=1(R8), X=1(R9), B=1(R12) -> 0x4F
        // ModRM: mod=00, reg=000(R8&7), rm=100(SIB)
        // SIB: scale=1(2x), index=001(R9&7), base=100(R12&7)
        assert_eq!(bytes, vec![0x4F, 0x8B, 0x04, 0x4C]);
    }

    #[test]
    fn test_mov_mr_sib_store() {
        // MOV [RBX + RCX*4], RAX: REX.W(48) + 89 + ModRM(00 000 100) + SIB
        let bytes = encode(
            X86Opcode::MovMRSib,
            &X86InstOperands {
                dst: Some(RAX), // src register stored in dst field for stores
                base: Some(RBX),
                index: Some(RCX),
                scale: 4,
                disp: 0,
                ..X86InstOperands::none()
            },
        );
        assert_eq!(bytes, vec![0x48, 0x89, 0x04, 0x8B]);
    }

    #[test]
    fn test_mov_mr_sib_store_disp8() {
        // MOV [RBX + RDX*8 + 32], RCX: REX.W(48) + 89 + ModRM(01 001 100) + SIB + disp8
        let bytes = encode(
            X86Opcode::MovMRSib,
            &X86InstOperands {
                dst: Some(RCX),
                base: Some(RBX),
                index: Some(RDX),
                scale: 8,
                disp: 32,
                ..X86InstOperands::none()
            },
        );
        assert_eq!(bytes, vec![0x48, 0x89, 0x4C, 0xD3, 0x20]);
    }

    // -----------------------------------------------------------------------
    // SSE conversion instruction tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_cvtsi2sd_xmm0_rax() {
        // CVTSI2SD XMM0, RAX: F2 REX.W(48) 0F 2A + ModRM(11 000 000)
        let bytes = encode(X86Opcode::Cvtsi2sd, &X86InstOperands::rr(XMM0, RAX));
        assert_eq!(bytes, vec![0xF2, 0x48, 0x0F, 0x2A, 0xC0]);
    }

    #[test]
    fn test_cvtsi2sd_xmm8_r15() {
        // CVTSI2SD XMM8, R15: F2 REX.WRB(4D) 0F 2A + ModRM(11 000 111)
        let bytes = encode(X86Opcode::Cvtsi2sd, &X86InstOperands::rr(XMM8, R15));
        assert_eq!(bytes, vec![0xF2, 0x4D, 0x0F, 0x2A, 0xC7]);
    }

    #[test]
    fn test_cvtsd2si_rax_xmm0() {
        // CVTSD2SI RAX, XMM0: F2 REX.W(48) 0F 2D + ModRM(11 000 000)
        let bytes = encode(X86Opcode::Cvtsd2si, &X86InstOperands::rr(RAX, XMM0));
        assert_eq!(bytes, vec![0xF2, 0x48, 0x0F, 0x2D, 0xC0]);
    }

    #[test]
    fn test_cvtsd2si_r8_xmm15() {
        // CVTSD2SI R8, XMM15: F2 REX.WRB(4D) 0F 2D + ModRM(11 000 111)
        let bytes = encode(X86Opcode::Cvtsd2si, &X86InstOperands::rr(R8, XMM15));
        assert_eq!(bytes, vec![0xF2, 0x4D, 0x0F, 0x2D, 0xC7]);
    }

    #[test]
    fn test_cvtsi2ss_xmm0_rax() {
        // CVTSI2SS XMM0, RAX: F3 REX.W(48) 0F 2A + ModRM(11 000 000)
        let bytes = encode(X86Opcode::Cvtsi2ss, &X86InstOperands::rr(XMM0, RAX));
        assert_eq!(bytes, vec![0xF3, 0x48, 0x0F, 0x2A, 0xC0]);
    }

    #[test]
    fn test_cvtss2si_rax_xmm0() {
        // CVTSS2SI RAX, XMM0: F3 REX.W(48) 0F 2D + ModRM(11 000 000)
        let bytes = encode(X86Opcode::Cvtss2si, &X86InstOperands::rr(RAX, XMM0));
        assert_eq!(bytes, vec![0xF3, 0x48, 0x0F, 0x2D, 0xC0]);
    }

    #[test]
    fn test_cvtsd2ss_xmm0_xmm1() {
        // CVTSD2SS XMM0, XMM1: F2 0F 5A + ModRM(11 000 001)
        let bytes = encode(X86Opcode::Cvtsd2ss, &X86InstOperands::rr(XMM0, XMM1));
        assert_eq!(bytes, vec![0xF2, 0x0F, 0x5A, 0xC1]);
    }

    #[test]
    fn test_cvtss2sd_xmm0_xmm1() {
        // CVTSS2SD XMM0, XMM1: F3 0F 5A + ModRM(11 000 001)
        let bytes = encode(X86Opcode::Cvtss2sd, &X86InstOperands::rr(XMM0, XMM1));
        assert_eq!(bytes, vec![0xF3, 0x0F, 0x5A, 0xC1]);
    }

    #[test]
    fn test_cvtsd2ss_xmm8_xmm15() {
        // CVTSD2SS XMM8, XMM15: F2 REX.RB(45) 0F 5A + ModRM(11 000 111)
        let bytes = encode(X86Opcode::Cvtsd2ss, &X86InstOperands::rr(XMM8, XMM15));
        assert_eq!(bytes, vec![0xF2, 0x45, 0x0F, 0x5A, 0xC7]);
    }

    #[test]
    fn test_cvtss2sd_xmm8_xmm15() {
        // CVTSS2SD XMM8, XMM15: F3 REX.RB(45) 0F 5A + ModRM(11 000 111)
        let bytes = encode(X86Opcode::Cvtss2sd, &X86InstOperands::rr(XMM8, XMM15));
        assert_eq!(bytes, vec![0xF3, 0x45, 0x0F, 0x5A, 0xC7]);
    }

    // -----------------------------------------------------------------------
    // New instruction size tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_new_instruction_sizes_v2() {
        let mut enc = X86Encoder::new();

        // LEA RIP: REX.W(1) + 8D(1) + ModRM(1) + disp32(4) = 7 bytes
        let n = enc.encode_instruction(X86Opcode::LeaRip, &X86InstOperands::rip_rel(RAX, 0)).unwrap();
        assert_eq!(n, 7);

        // CVTSI2SD: F2(1) + REX.W(1) + 0F(1) + 2A(1) + ModRM(1) = 5 bytes
        let n = enc.encode_instruction(X86Opcode::Cvtsi2sd, &X86InstOperands::rr(XMM0, RAX)).unwrap();
        assert_eq!(n, 5);

        // CVTSD2SI: F2(1) + REX.W(1) + 0F(1) + 2D(1) + ModRM(1) = 5 bytes
        let n = enc.encode_instruction(X86Opcode::Cvtsd2si, &X86InstOperands::rr(RAX, XMM0)).unwrap();
        assert_eq!(n, 5);

        // CVTSD2SS: F2(1) + 0F(1) + 5A(1) + ModRM(1) = 4 bytes
        let n = enc.encode_instruction(X86Opcode::Cvtsd2ss, &X86InstOperands::rr(XMM0, XMM1)).unwrap();
        assert_eq!(n, 4);

        // CVTSS2SD: F3(1) + 0F(1) + 5A(1) + ModRM(1) = 4 bytes
        let n = enc.encode_instruction(X86Opcode::Cvtss2sd, &X86InstOperands::rr(XMM0, XMM1)).unwrap();
        assert_eq!(n, 4);

        // MovRMSib [base + index*scale]: REX.W(1) + 8B(1) + ModRM(1) + SIB(1) = 4 bytes
        let n = enc.encode_instruction(
            X86Opcode::MovRMSib,
            &X86InstOperands::rm_sib(RAX, RBX, RCX, 4, 0),
        ).unwrap();
        assert_eq!(n, 4);

        // MovRMSib [base + index*scale + disp8]: 5 bytes
        let n = enc.encode_instruction(
            X86Opcode::MovRMSib,
            &X86InstOperands::rm_sib(RAX, RBX, RCX, 4, 16),
        ).unwrap();
        assert_eq!(n, 5);

        // MovRMSib [base + index*scale + disp32]: 8 bytes
        let n = enc.encode_instruction(
            X86Opcode::MovRMSib,
            &X86InstOperands::rm_sib(RAX, RBX, RCX, 8, 256),
        ).unwrap();
        assert_eq!(n, 8);
    }

    // -----------------------------------------------------------------------
    // SIB error handling
    // -----------------------------------------------------------------------

    #[test]
    fn test_mov_rm_sib_missing_base_error() {
        let mut enc = X86Encoder::new();
        let ops = X86InstOperands {
            dst: Some(RAX),
            index: Some(RCX),
            scale: 4,
            ..X86InstOperands::none()
        };
        let result = enc.encode_instruction(X86Opcode::MovRMSib, &ops);
        assert!(result.is_err());
    }

    #[test]
    fn test_mov_rm_sib_missing_index_error() {
        let mut enc = X86Encoder::new();
        let ops = X86InstOperands {
            dst: Some(RAX),
            base: Some(RBX),
            scale: 4,
            ..X86InstOperands::none()
        };
        let result = enc.encode_instruction(X86Opcode::MovRMSib, &ops);
        assert!(result.is_err());
    }

    #[test]
    fn test_lea_rip_missing_dst_error() {
        let mut enc = X86Encoder::new();
        let result = enc.encode_instruction(X86Opcode::LeaRip, &X86InstOperands::none());
        assert!(result.is_err());
    }

    // ===================================================================
    // Cross-reference encoding verification
    //
    // These tests systematically verify byte-level encoding against the
    // Intel 64 and IA-32 Architectures SDM Volume 2 (Instruction Set
    // Reference). Each test cites the relevant SDM instruction family.
    // ===================================================================

    // -------------------------------------------------------------------
    // 1. MOV reg,reg for all 16 GPRs -- verify REX bits
    // Intel SDM Vol 2, MOV instruction: opcode 89 /r (MOV r/m64, r64)
    // REX prefix 0100_WRXB: W=1 (64-bit), R=src>>3, B=dst>>3
    // ModRM = 11_src[2:0]_dst[2:0]
    // -------------------------------------------------------------------

    #[test]
    fn test_mov_all_16_gprs() {
        // Pairs: (dst, src, expected_bytes)
        // Format: REX.W prefix + 0x89 + ModRM(11, src[2:0], dst[2:0])
        let cases: &[(X86PReg, X86PReg, &[u8])] = &[
            // Legacy-to-legacy (no extended regs): REX.W = 0x48
            (RAX, RAX, &[0x48, 0x89, 0xC0]),  // MOV RAX,RAX: ModRM=11_000_000=C0
            (RCX, RAX, &[0x48, 0x89, 0xC1]),  // MOV RCX,RAX: ModRM=11_000_001=C1
            (RDX, RBX, &[0x48, 0x89, 0xDA]),  // MOV RDX,RBX: ModRM=11_011_010=DA
            (RSP, RBP, &[0x48, 0x89, 0xEC]),  // MOV RSP,RBP: ModRM=11_101_100=EC
            (RSI, RDI, &[0x48, 0x89, 0xFE]),  // MOV RSI,RDI: ModRM=11_111_110=FE
            (RDI, RSI, &[0x48, 0x89, 0xF7]),  // MOV RDI,RSI: ModRM=11_110_111=F7

            // Extended dst only: REX.WB = 0x49 (W=1, B=1)
            (R8,  RAX, &[0x49, 0x89, 0xC0]),  // MOV R8,RAX:  ModRM=11_000_000=C0
            (R12, RCX, &[0x49, 0x89, 0xCC]),  // MOV R12,RCX: ModRM=11_001_100=CC
            (R15, RDI, &[0x49, 0x89, 0xFF]),  // MOV R15,RDI: ModRM=11_111_111=FF

            // Extended src only: REX.WR = 0x4C (W=1, R=1)
            (RAX, R8,  &[0x4C, 0x89, 0xC0]),  // MOV RAX,R8:  ModRM=11_000_000=C0
            (RCX, R12, &[0x4C, 0x89, 0xE1]),  // MOV RCX,R12: ModRM=11_100_001=E1
            (RDI, R15, &[0x4C, 0x89, 0xFF]),  // MOV RDI,R15: ModRM=11_111_111=FF

            // Both extended: REX.WRB = 0x4D (W=1, R=1, B=1)
            (R8,  R9,  &[0x4D, 0x89, 0xC8]),  // MOV R8,R9:   ModRM=11_001_000=C8
            (R10, R11, &[0x4D, 0x89, 0xDA]),  // MOV R10,R11: ModRM=11_011_010=DA
            (R14, R13, &[0x4D, 0x89, 0xEE]),  // MOV R14,R13: ModRM=11_101_110=EE
            (R15, R15, &[0x4D, 0x89, 0xFF]),  // MOV R15,R15: ModRM=11_111_111=FF
        ];

        for (i, (dst, src, expected)) in cases.iter().enumerate() {
            let bytes = encode(X86Opcode::MovRR, &X86InstOperands::rr(*dst, *src));
            assert_eq!(
                bytes, expected.to_vec(),
                "MOV case {}: dst={:?}, src={:?}", i, dst, src
            );
        }
    }

    // -------------------------------------------------------------------
    // 2. PUSH/POP for all 16 GPRs
    // Intel SDM Vol 2, PUSH: opcode 50+rd (no REX.W needed for PUSH r64)
    //   R8-R15 need REX.B prefix (0x41) to extend the opcode register field
    // Intel SDM Vol 2, POP:  opcode 58+rd (same REX.B rule)
    // -------------------------------------------------------------------

    #[test]
    fn test_push_pop_all_16_gprs() {
        // PUSH register tests: (reg, expected_bytes)
        let push_cases: &[(X86PReg, &[u8])] = &[
            (RAX, &[0x50]),          // 50+0
            (RCX, &[0x51]),          // 50+1
            (RDX, &[0x52]),          // 50+2
            (RBX, &[0x53]),          // 50+3
            (RSP, &[0x54]),          // 50+4
            (RBP, &[0x55]),          // 50+5
            (RSI, &[0x56]),          // 50+6
            (RDI, &[0x57]),          // 50+7
            (R8,  &[0x41, 0x50]),    // REX.B + 50+0
            (R9,  &[0x41, 0x51]),    // REX.B + 50+1
            (R10, &[0x41, 0x52]),    // REX.B + 50+2
            (R11, &[0x41, 0x53]),    // REX.B + 50+3
            (R12, &[0x41, 0x54]),    // REX.B + 50+4
            (R13, &[0x41, 0x55]),    // REX.B + 50+5
            (R14, &[0x41, 0x56]),    // REX.B + 50+6
            (R15, &[0x41, 0x57]),    // REX.B + 50+7
        ];

        for (i, (reg, expected)) in push_cases.iter().enumerate() {
            let bytes = encode(X86Opcode::Push, &X86InstOperands::r(*reg));
            assert_eq!(
                bytes, expected.to_vec(),
                "PUSH case {}: reg={:?}", i, reg
            );
        }

        // POP register tests: (reg, expected_bytes)
        let pop_cases: &[(X86PReg, &[u8])] = &[
            (RAX, &[0x58]),          // 58+0
            (RCX, &[0x59]),          // 58+1
            (RDX, &[0x5A]),          // 58+2
            (RBX, &[0x5B]),          // 58+3
            (RSP, &[0x5C]),          // 58+4
            (RBP, &[0x5D]),          // 58+5
            (RSI, &[0x5E]),          // 58+6
            (RDI, &[0x5F]),          // 58+7
            (R8,  &[0x41, 0x58]),    // REX.B + 58+0
            (R9,  &[0x41, 0x59]),    // REX.B + 58+1
            (R10, &[0x41, 0x5A]),    // REX.B + 58+2
            (R11, &[0x41, 0x5B]),    // REX.B + 58+3
            (R12, &[0x41, 0x5C]),    // REX.B + 58+4
            (R13, &[0x41, 0x5D]),    // REX.B + 58+5
            (R14, &[0x41, 0x5E]),    // REX.B + 58+6
            (R15, &[0x41, 0x5F]),    // REX.B + 58+7
        ];

        for (i, (reg, expected)) in pop_cases.iter().enumerate() {
            let bytes = encode(X86Opcode::Pop, &X86InstOperands::r(*reg));
            assert_eq!(
                bytes, expected.to_vec(),
                "POP case {}: reg={:?}", i, reg
            );
        }
    }

    // -------------------------------------------------------------------
    // 3. Negative displacement: MOV RAX, [RBX-8]
    // Intel SDM Vol 2, MOV: opcode 8B /r (MOV r64, r/m64)
    // disp=-8 fits in signed byte (0xF8), so mod=01 (disp8)
    // -------------------------------------------------------------------

    #[test]
    fn test_negative_displacement() {
        // MOV RAX, [RBX-8]: REX.W(48) + 8B + ModRM(01_000_011=43) + disp8(F8)
        // -8 as signed byte = 0xF8
        let bytes = encode(X86Opcode::MovRM, &X86InstOperands::rm(RAX, RBX, -8));
        assert_eq!(bytes, vec![0x48, 0x8B, 0x43, 0xF8]);

        // MOV RCX, [RDX-1]: REX.W(48) + 8B + ModRM(01_001_010=4A) + disp8(FF)
        let bytes = encode(X86Opcode::MovRM, &X86InstOperands::rm(RCX, RDX, -1));
        assert_eq!(bytes, vec![0x48, 0x8B, 0x4A, 0xFF]);

        // MOV RDI, [RSI-128]: disp=-128 still fits in disp8 (0x80)
        // REX.W(48) + 8B + ModRM(01_111_110=7E) + disp8(80)
        let bytes = encode(X86Opcode::MovRM, &X86InstOperands::rm(RDI, RSI, -128));
        assert_eq!(bytes, vec![0x48, 0x8B, 0x7E, 0x80]);

        // MOV RAX, [RBX-129]: disp=-129 does NOT fit in disp8, needs disp32
        // REX.W(48) + 8B + ModRM(10_000_011=83) + disp32(FFFFFF7F in LE)
        let bytes = encode(X86Opcode::MovRM, &X86InstOperands::rm(RAX, RBX, -129));
        assert_eq!(bytes, vec![0x48, 0x8B, 0x83, 0x7F, 0xFF, 0xFF, 0xFF]);
    }

    // -------------------------------------------------------------------
    // 4. RSP and RBP base addressing special cases
    // Intel SDM Vol 2, Table 2-2 (ModRM with SIB):
    //   rm=100 (RSP/R12) always emits SIB byte
    //   rm=101 (RBP/R13) with mod=00 is RIP-relative, so disp=0 needs mod=01+disp8(00)
    // -------------------------------------------------------------------

    #[test]
    fn test_rsp_rbp_addressing() {
        // MOV RAX, [RSP+0]: needs SIB byte (base=RSP, no index)
        // REX.W(48) + 8B + ModRM(00_000_100=04) + SIB(00_100_100=24)
        let bytes = encode(X86Opcode::MovRM, &X86InstOperands::rm(RAX, RSP, 0));
        assert_eq!(bytes, vec![0x48, 0x8B, 0x04, 0x24]);

        // MOV RAX, [RSP+8]: SIB + disp8
        // REX.W(48) + 8B + ModRM(01_000_100=44) + SIB(00_100_100=24) + disp8(08)
        let bytes = encode(X86Opcode::MovRM, &X86InstOperands::rm(RAX, RSP, 8));
        assert_eq!(bytes, vec![0x48, 0x8B, 0x44, 0x24, 0x08]);

        // MOV RAX, [RSP+256]: SIB + disp32
        // REX.W(48) + 8B + ModRM(10_000_100=84) + SIB(00_100_100=24) + disp32
        let bytes = encode(X86Opcode::MovRM, &X86InstOperands::rm(RAX, RSP, 256));
        assert_eq!(bytes, vec![0x48, 0x8B, 0x84, 0x24, 0x00, 0x01, 0x00, 0x00]);

        // MOV RAX, [RBP+0]: RBP base with no displacement needs disp8=0
        // REX.W(48) + 8B + ModRM(01_000_101=45) + disp8(00)
        let bytes = encode(X86Opcode::MovRM, &X86InstOperands::rm(RAX, RBP, 0));
        assert_eq!(bytes, vec![0x48, 0x8B, 0x45, 0x00]);

        // MOV RAX, [RBP+16]: normal disp8
        // REX.W(48) + 8B + ModRM(01_000_101=45) + disp8(10)
        let bytes = encode(X86Opcode::MovRM, &X86InstOperands::rm(RAX, RBP, 16));
        assert_eq!(bytes, vec![0x48, 0x8B, 0x45, 0x10]);

        // MOV R8, [R12+0]: R12 (hw=4) behaves like RSP -- needs SIB
        // REX.WRB(4D) + 8B + ModRM(00_000_100=04) + SIB(00_100_100=24)
        let bytes = encode(X86Opcode::MovRM, &X86InstOperands::rm(R8, R12, 0));
        assert_eq!(bytes, vec![0x4D, 0x8B, 0x04, 0x24]);

        // MOV R8, [R13+0]: R13 (hw=5) behaves like RBP -- needs disp8=0
        // REX.WRB(4D) + 8B + ModRM(01_000_101=45) + disp8(00)
        let bytes = encode(X86Opcode::MovRM, &X86InstOperands::rm(R8, R13, 0));
        assert_eq!(bytes, vec![0x4D, 0x8B, 0x45, 0x00]);
    }

    // -------------------------------------------------------------------
    // 5. Jcc condition codes -- verify 0F 80+cc rel32 encoding
    // Intel SDM Vol 2, Jcc: 0F 80+cc cd (near jump with 32-bit displacement)
    // cc values: O=0, NO=1, B=2, AE=3, E=4, NE=5, BE=6, A=7,
    //            S=8, NS=9, P=A, NP=B, L=C, GE=D, LE=E, G=F
    // -------------------------------------------------------------------

    #[test]
    fn test_all_jcc_condition_codes() {
        // Each Jcc with disp=0 should produce: 0F (80+cc) 00 00 00 00
        let cases: &[(X86CondCode, u8)] = &[
            (X86CondCode::O,  0x80),
            (X86CondCode::NO, 0x81),
            (X86CondCode::B,  0x82),
            (X86CondCode::AE, 0x83),
            (X86CondCode::E,  0x84),
            (X86CondCode::NE, 0x85),
            (X86CondCode::BE, 0x86),
            (X86CondCode::A,  0x87),
            (X86CondCode::S,  0x88),
            (X86CondCode::NS, 0x89),
            (X86CondCode::P,  0x8A),
            (X86CondCode::NP, 0x8B),
            (X86CondCode::L,  0x8C),
            (X86CondCode::GE, 0x8D),
            (X86CondCode::LE, 0x8E),
            (X86CondCode::G,  0x8F),
        ];

        for (cc, expected_opcode2) in cases {
            let bytes = encode(
                X86Opcode::Jcc,
                &X86InstOperands::jcc(*cc, 0),
            );
            assert_eq!(bytes.len(), 6, "Jcc {:?} should be 6 bytes", cc);
            assert_eq!(bytes[0], 0x0F, "Jcc {:?} first byte", cc);
            assert_eq!(bytes[1], *expected_opcode2, "Jcc {:?} second byte", cc);
            // disp32=0 -> 00 00 00 00
            assert_eq!(&bytes[2..], &[0x00, 0x00, 0x00, 0x00], "Jcc {:?} disp", cc);
        }

        // Verify a nonzero displacement: JNE +100
        // 0F 85 64 00 00 00  (100 = 0x64)
        let bytes = encode(
            X86Opcode::Jcc,
            &X86InstOperands::jcc(X86CondCode::NE, 100),
        );
        assert_eq!(bytes, vec![0x0F, 0x85, 0x64, 0x00, 0x00, 0x00]);

        // Verify negative displacement: JL -16
        // 0F 8C F0 FF FF FF  (-16 as i32 = 0xFFFF_FFF0)
        let bytes = encode(
            X86Opcode::Jcc,
            &X86InstOperands::jcc(X86CondCode::L, -16),
        );
        assert_eq!(bytes, vec![0x0F, 0x8C, 0xF0, 0xFF, 0xFF, 0xFF]);
    }

    // -------------------------------------------------------------------
    // 6. XOR reg,reg zero idiom
    // Intel SDM Vol 2, XOR: opcode 31 /r (XOR r/m64, r64)
    // REX.W(48) + 31 + ModRM(11, src[2:0], dst[2:0])
    // -------------------------------------------------------------------

    #[test]
    fn test_xor_zero_idiom() {
        // XOR RAX, RAX: REX.W(48) + 31 + ModRM(11_000_000=C0)
        let bytes = encode(X86Opcode::XorRR, &X86InstOperands::rr(RAX, RAX));
        assert_eq!(bytes, vec![0x48, 0x31, 0xC0]);

        // XOR RCX, RCX: REX.W(48) + 31 + ModRM(11_001_001=C9)
        let bytes = encode(X86Opcode::XorRR, &X86InstOperands::rr(RCX, RCX));
        assert_eq!(bytes, vec![0x48, 0x31, 0xC9]);

        // XOR RDX, RDX: REX.W(48) + 31 + ModRM(11_010_010=D2)
        let bytes = encode(X86Opcode::XorRR, &X86InstOperands::rr(RDX, RDX));
        assert_eq!(bytes, vec![0x48, 0x31, 0xD2]);

        // XOR R8, R8: REX.WRB(4D) + 31 + ModRM(11_000_000=C0)
        let bytes = encode(X86Opcode::XorRR, &X86InstOperands::rr(R8, R8));
        assert_eq!(bytes, vec![0x4D, 0x31, 0xC0]);

        // XOR R15, R15: REX.WRB(4D) + 31 + ModRM(11_111_111=FF)
        let bytes = encode(X86Opcode::XorRR, &X86InstOperands::rr(R15, R15));
        assert_eq!(bytes, vec![0x4D, 0x31, 0xFF]);

        // XOR R10, R10: REX.WRB(4D) + 31 + ModRM(11_010_010=D2)
        let bytes = encode(X86Opcode::XorRR, &X86InstOperands::rr(R10, R10));
        assert_eq!(bytes, vec![0x4D, 0x31, 0xD2]);
    }

    // -------------------------------------------------------------------
    // 7. SUB RSP, imm32 -- common prologue stack frame allocation
    // Intel SDM Vol 2, SUB: opcode 81 /5 id (SUB r/m64, imm32)
    // REX.W(48) + 81 + ModRM(11_101_100=EC) + imm32
    // RSP hw_enc=4, /5 means reg field=5 in ModRM
    // -------------------------------------------------------------------

    #[test]
    fn test_sub_rsp_imm32() {
        // SUB RSP, 32:  REX.W(48) + 81 + ModRM(11_101_100=EC) + imm32(20 00 00 00)
        let bytes = encode(X86Opcode::SubRI, &X86InstOperands::ri(RSP, 32));
        assert_eq!(bytes, vec![0x48, 0x81, 0xEC, 0x20, 0x00, 0x00, 0x00]);

        // SUB RSP, 128: REX.W(48) + 81 + EC + imm32(80 00 00 00)
        let bytes = encode(X86Opcode::SubRI, &X86InstOperands::ri(RSP, 128));
        assert_eq!(bytes, vec![0x48, 0x81, 0xEC, 0x80, 0x00, 0x00, 0x00]);

        // ADD RSP, 32 (epilogue counterpart):
        // REX.W(48) + 81 + ModRM(11_000_100=C4) + imm32(20 00 00 00)
        // /0 means reg field=0 in ModRM for ADD
        let bytes = encode(X86Opcode::AddRI, &X86InstOperands::ri(RSP, 32));
        assert_eq!(bytes, vec![0x48, 0x81, 0xC4, 0x20, 0x00, 0x00, 0x00]);
    }

    // -------------------------------------------------------------------
    // 8. Complete prologue/epilogue sequence
    // Verifies concatenated bytes for a standard System V AMD64 ABI
    // function frame setup and teardown.
    //
    //   PUSH RBP           ; 55
    //   MOV RBP, RSP       ; 48 89 E5 (REX.W + 89 + ModRM 11_100_101)
    //   SUB RSP, 32        ; 48 81 EC 20 00 00 00
    //   ADD RSP, 32        ; 48 81 C4 20 00 00 00
    //   POP RBP            ; 5D
    //   RET                ; C3
    //
    // Total: 1 + 3 + 7 + 7 + 1 + 1 = 20 bytes
    // -------------------------------------------------------------------

    #[test]
    fn test_prologue_epilogue_sequence() {
        let mut enc = X86Encoder::new();

        // PUSH RBP
        enc.encode_instruction(X86Opcode::Push, &X86InstOperands::r(RBP)).unwrap();
        // MOV RBP, RSP
        enc.encode_instruction(X86Opcode::MovRR, &X86InstOperands::rr(RBP, RSP)).unwrap();
        // SUB RSP, 32
        enc.encode_instruction(X86Opcode::SubRI, &X86InstOperands::ri(RSP, 32)).unwrap();
        // ADD RSP, 32
        enc.encode_instruction(X86Opcode::AddRI, &X86InstOperands::ri(RSP, 32)).unwrap();
        // POP RBP
        enc.encode_instruction(X86Opcode::Pop, &X86InstOperands::r(RBP)).unwrap();
        // RET
        enc.encode_instruction(X86Opcode::Ret, &X86InstOperands::none()).unwrap();

        let bytes = enc.finish();

        let expected: Vec<u8> = vec![
            0x55,                               // PUSH RBP
            0x48, 0x89, 0xE5,                   // MOV RBP, RSP
            0x48, 0x81, 0xEC, 0x20, 0x00, 0x00, 0x00, // SUB RSP, 32
            0x48, 0x81, 0xC4, 0x20, 0x00, 0x00, 0x00, // ADD RSP, 32
            0x5D,                               // POP RBP
            0xC3,                               // RET
        ];

        assert_eq!(bytes.len(), 20, "prologue/epilogue should be 20 bytes");
        assert_eq!(bytes, expected);
    }

    // -----------------------------------------------------------------------
    // MovzxW tests (MOVZX r64, r/m16 -- 0F B7)
    // -----------------------------------------------------------------------

    #[test]
    fn test_movzxw_rax_cx() {
        // MOVZX RAX, CX: REX.W(48) + 0F B7 + ModRM(11 000 001)
        let bytes = encode(X86Opcode::MovzxW, &X86InstOperands::rr(RAX, CX));
        assert_eq!(bytes, vec![0x48, 0x0F, 0xB7, 0xC1]);
    }

    #[test]
    fn test_movzxw_r8_ax() {
        // MOVZX R8, AX: REX.WR(4C) + 0F B7 + ModRM(11 000 000)
        let bytes = encode(X86Opcode::MovzxW, &X86InstOperands::rr(R8, AX));
        assert_eq!(bytes, vec![0x4C, 0x0F, 0xB7, 0xC0]);
    }

    // -----------------------------------------------------------------------
    // MovsxB tests (MOVSX r64, r/m8 -- 0F BE)
    // -----------------------------------------------------------------------

    #[test]
    fn test_movsxb_rax_cl() {
        // MOVSX RAX, CL: REX.W(48) + 0F BE + ModRM(11 000 001)
        let bytes = encode(X86Opcode::MovsxB, &X86InstOperands::rr(RAX, CL));
        assert_eq!(bytes, vec![0x48, 0x0F, 0xBE, 0xC1]);
    }

    #[test]
    fn test_movsxb_r8_al() {
        // MOVSX R8, AL: REX.WR(4C) + 0F BE + ModRM(11 000 000)
        let bytes = encode(X86Opcode::MovsxB, &X86InstOperands::rr(R8, AL));
        assert_eq!(bytes, vec![0x4C, 0x0F, 0xBE, 0xC0]);
    }

    // -----------------------------------------------------------------------
    // MovsxW tests (MOVSX r64, r/m16 -- 0F BF)
    // -----------------------------------------------------------------------

    #[test]
    fn test_movsxw_rax_cx() {
        // MOVSX RAX, CX: REX.W(48) + 0F BF + ModRM(11 000 001)
        let bytes = encode(X86Opcode::MovsxW, &X86InstOperands::rr(RAX, CX));
        assert_eq!(bytes, vec![0x48, 0x0F, 0xBF, 0xC1]);
    }

    #[test]
    fn test_movsxw_r15_r14w() {
        // MOVSX R15, R14W: REX.WRB(4D) + 0F BF + ModRM(11 111 110)
        let bytes = encode(X86Opcode::MovsxW, &X86InstOperands::rr(R15, R14W));
        assert_eq!(bytes, vec![0x4D, 0x0F, 0xBF, 0xFE]);
    }

    // -----------------------------------------------------------------------
    // LeaSib tests (LEA r64, [base + index*scale + disp])
    // -----------------------------------------------------------------------

    #[test]
    fn test_lea_sib_rax_rbx_rcx_scale4() {
        // LEA RAX, [RBX + RCX*4]: REX.W(48) + 8D + ModRM(00 000 100) + SIB(10 001 011)
        let bytes = encode(
            X86Opcode::LeaSib,
            &X86InstOperands::rm_sib(RAX, RBX, RCX, 4, 0),
        );
        assert_eq!(bytes, vec![0x48, 0x8D, 0x04, 0x8B]);
    }

    #[test]
    fn test_lea_sib_rax_rbx_rcx_scale4_disp8() {
        // LEA RAX, [RBX + RCX*4 + 16]: REX.W(48) + 8D + ModRM(01 000 100) + SIB(10 001 011) + disp8(10)
        let bytes = encode(
            X86Opcode::LeaSib,
            &X86InstOperands::rm_sib(RAX, RBX, RCX, 4, 16),
        );
        assert_eq!(bytes, vec![0x48, 0x8D, 0x44, 0x8B, 0x10]);
    }

    #[test]
    fn test_lea_sib_r8_r12_r9_scale2() {
        // LEA R8, [R12 + R9*2]: REX.WRXB(4F) + 8D + ModRM(00 000 100) + SIB(01 001 100)
        let bytes = encode(
            X86Opcode::LeaSib,
            &X86InstOperands::rm_sib(R8, R12, R9, 2, 0),
        );
        assert_eq!(bytes, vec![0x4F, 0x8D, 0x04, 0x4C]);
    }

    // -----------------------------------------------------------------------
    // ImulRM tests (IMUL r64, [base+disp] -- 0F AF)
    // -----------------------------------------------------------------------

    #[test]
    fn test_imul_rax_mem_rbx() {
        // IMUL RAX, [RBX]: REX.W(48) + 0F AF + ModRM(00 000 011)
        let bytes = encode(X86Opcode::ImulRM, &X86InstOperands::rm(RAX, RBX, 0));
        assert_eq!(bytes, vec![0x48, 0x0F, 0xAF, 0x03]);
    }

    #[test]
    fn test_imul_rax_mem_rbx_disp8() {
        // IMUL RAX, [RBX+16]: REX.W(48) + 0F AF + ModRM(01 000 011) + disp8(10)
        let bytes = encode(X86Opcode::ImulRM, &X86InstOperands::rm(RAX, RBX, 16));
        assert_eq!(bytes, vec![0x48, 0x0F, 0xAF, 0x43, 0x10]);
    }

    #[test]
    fn test_imul_r8_mem_r13() {
        // IMUL R8, [R13+0]: REX.WRB(4D) + 0F AF + ModRM(01 000 101) + disp8(00)
        // R13 base (hw_enc & 7 == 5) with disp=0 requires disp8=0
        let bytes = encode(X86Opcode::ImulRM, &X86InstOperands::rm(R8, R13, 0));
        assert_eq!(bytes, vec![0x4D, 0x0F, 0xAF, 0x45, 0x00]);
    }

    // -----------------------------------------------------------------------
    // TestRM tests (TEST r64, [base+disp] -- 85)
    // -----------------------------------------------------------------------

    #[test]
    fn test_test_rax_mem_rbx() {
        // TEST RAX, [RBX]: REX.W(48) + 85 + ModRM(00 000 011)
        let bytes = encode(X86Opcode::TestRM, &X86InstOperands::rm(RAX, RBX, 0));
        assert_eq!(bytes, vec![0x48, 0x85, 0x03]);
    }

    #[test]
    fn test_test_rcx_mem_rdx_disp8() {
        // TEST RCX, [RDX+16]: REX.W(48) + 85 + ModRM(01 001 010) + disp8(10)
        let bytes = encode(X86Opcode::TestRM, &X86InstOperands::rm(RCX, RDX, 16));
        assert_eq!(bytes, vec![0x48, 0x85, 0x4A, 0x10]);
    }

    // -----------------------------------------------------------------------
    // CallM tests (CALL [base+disp] -- FF /2)
    // -----------------------------------------------------------------------

    #[test]
    fn test_call_mem_rax() {
        // CALL [RAX]: FF + ModRM(00 010 000) = 0x10
        let bytes = encode(
            X86Opcode::CallM,
            &X86InstOperands {
                base: Some(RAX),
                disp: 0,
                ..X86InstOperands::none()
            },
        );
        assert_eq!(bytes, vec![0xFF, 0x10]);
    }

    #[test]
    fn test_call_mem_rbx_disp8() {
        // CALL [RBX+8]: FF + ModRM(01 010 011) + disp8(08)
        let bytes = encode(
            X86Opcode::CallM,
            &X86InstOperands {
                base: Some(RBX),
                disp: 8,
                ..X86InstOperands::none()
            },
        );
        assert_eq!(bytes, vec![0xFF, 0x53, 0x08]);
    }

    #[test]
    fn test_call_mem_r15() {
        // CALL [R15]: REX.B(41) + FF + ModRM(00 010 111)
        let bytes = encode(
            X86Opcode::CallM,
            &X86InstOperands {
                base: Some(R15),
                disp: 0,
                ..X86InstOperands::none()
            },
        );
        assert_eq!(bytes, vec![0x41, 0xFF, 0x17]);
    }

    #[test]
    fn test_call_mem_rsp() {
        // CALL [RSP]: FF + ModRM(00 010 100) + SIB(00 100 100)
        // RSP base requires SIB byte
        let bytes = encode(
            X86Opcode::CallM,
            &X86InstOperands {
                base: Some(RSP),
                disp: 0,
                ..X86InstOperands::none()
            },
        );
        assert_eq!(bytes, vec![0xFF, 0x14, 0x24]);
    }

    // -----------------------------------------------------------------------
    // CMP r/m64, imm8 (CmpRI8) tests
    // Intel SDM Vol 2: CMP r/m64, imm8: REX.W + 83 /7 ib
    // -----------------------------------------------------------------------

    #[test]
    fn test_cmp_rax_imm8() {
        // CMP RAX, 1: REX.W(48) + 83 + ModRM(11 111 000) + imm8(01)
        let bytes = encode(X86Opcode::CmpRI8, &X86InstOperands::ri(RAX, 1));
        assert_eq!(bytes, vec![0x48, 0x83, 0xF8, 0x01]);
    }

    #[test]
    fn test_cmp_rcx_imm8_negative() {
        // CMP RCX, -1: REX.W(48) + 83 + ModRM(11 111 001) + imm8(FF)
        let bytes = encode(X86Opcode::CmpRI8, &X86InstOperands::ri(RCX, -1));
        assert_eq!(bytes, vec![0x48, 0x83, 0xF9, 0xFF]);
    }

    #[test]
    fn test_cmp_r15_imm8() {
        // CMP R15, 42: REX.WB(49) + 83 + ModRM(11 111 111) + imm8(2A)
        let bytes = encode(X86Opcode::CmpRI8, &X86InstOperands::ri(R15, 42));
        assert_eq!(bytes, vec![0x49, 0x83, 0xFF, 0x2A]);
    }

    #[test]
    fn test_cmp_r8_imm8_zero() {
        // CMP R8, 0: REX.WB(49) + 83 + ModRM(11 111 000) + imm8(00)
        let bytes = encode(X86Opcode::CmpRI8, &X86InstOperands::ri(R8, 0));
        assert_eq!(bytes, vec![0x49, 0x83, 0xF8, 0x00]);
    }

    #[test]
    fn test_cmpri8_vs_cmpri_size() {
        // CmpRI8 (short form) should be 4 bytes vs CmpRI's 7 bytes
        let mut enc = X86Encoder::new();
        let n8 = enc.encode_instruction(X86Opcode::CmpRI8, &X86InstOperands::ri(RAX, 1)).unwrap();
        assert_eq!(n8, 4); // REX.W + 83 + ModRM + imm8
        let n32 = enc.encode_instruction(X86Opcode::CmpRI, &X86InstOperands::ri(RAX, 1)).unwrap();
        assert_eq!(n32, 7); // REX.W + 81 + ModRM + imm32
    }

    // -----------------------------------------------------------------------
    // CDQ / CQO tests
    // Intel SDM Vol 2: CDQ: 99, CQO: REX.W + 99
    // -----------------------------------------------------------------------

    #[test]
    fn test_cdq() {
        // CDQ: 99 (sign-extend EAX into EDX:EAX, 32-bit)
        let bytes = encode(X86Opcode::Cdq, &X86InstOperands::none());
        assert_eq!(bytes, vec![0x99]);
    }

    #[test]
    fn test_cqo() {
        // CQO: REX.W(48) + 99 (sign-extend RAX into RDX:RAX, 64-bit)
        let bytes = encode(X86Opcode::Cqo, &X86InstOperands::none());
        assert_eq!(bytes, vec![0x48, 0x99]);
    }

    #[test]
    fn test_cdq_cqo_sizes() {
        let mut enc = X86Encoder::new();
        let n = enc.encode_instruction(X86Opcode::Cdq, &X86InstOperands::none()).unwrap();
        assert_eq!(n, 1);
        let n = enc.encode_instruction(X86Opcode::Cqo, &X86InstOperands::none()).unwrap();
        assert_eq!(n, 2);
    }

    // -----------------------------------------------------------------------
    // Multi-byte NOP tests
    // Intel SDM Vol 2B, NOP instruction, Table 4-12
    // -----------------------------------------------------------------------

    #[test]
    fn test_multibyte_nop_0() {
        let mut enc = X86Encoder::new();
        enc.encode_multibyte_nop(0);
        assert_eq!(enc.finish(), vec![] as Vec<u8>);
    }

    #[test]
    fn test_multibyte_nop_1() {
        let mut enc = X86Encoder::new();
        enc.encode_multibyte_nop(1);
        assert_eq!(enc.finish(), vec![0x90]);
    }

    #[test]
    fn test_multibyte_nop_2() {
        let mut enc = X86Encoder::new();
        enc.encode_multibyte_nop(2);
        assert_eq!(enc.finish(), vec![0x66, 0x90]);
    }

    #[test]
    fn test_multibyte_nop_3() {
        let mut enc = X86Encoder::new();
        enc.encode_multibyte_nop(3);
        assert_eq!(enc.finish(), vec![0x0F, 0x1F, 0x00]);
    }

    #[test]
    fn test_multibyte_nop_4() {
        let mut enc = X86Encoder::new();
        enc.encode_multibyte_nop(4);
        assert_eq!(enc.finish(), vec![0x0F, 0x1F, 0x40, 0x00]);
    }

    #[test]
    fn test_multibyte_nop_5() {
        let mut enc = X86Encoder::new();
        enc.encode_multibyte_nop(5);
        assert_eq!(enc.finish(), vec![0x0F, 0x1F, 0x44, 0x00, 0x00]);
    }

    #[test]
    fn test_multibyte_nop_6() {
        let mut enc = X86Encoder::new();
        enc.encode_multibyte_nop(6);
        assert_eq!(enc.finish(), vec![0x66, 0x0F, 0x1F, 0x44, 0x00, 0x00]);
    }

    #[test]
    fn test_multibyte_nop_7() {
        let mut enc = X86Encoder::new();
        enc.encode_multibyte_nop(7);
        assert_eq!(enc.finish(), vec![0x0F, 0x1F, 0x80, 0x00, 0x00, 0x00, 0x00]);
    }

    #[test]
    fn test_multibyte_nop_8() {
        let mut enc = X86Encoder::new();
        enc.encode_multibyte_nop(8);
        assert_eq!(enc.finish(), vec![0x0F, 0x1F, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00]);
    }

    #[test]
    fn test_multibyte_nop_9() {
        let mut enc = X86Encoder::new();
        enc.encode_multibyte_nop(9);
        assert_eq!(enc.finish(), vec![0x66, 0x0F, 0x1F, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00]);
    }

    #[test]
    fn test_multibyte_nop_11_recurse() {
        // 11 bytes = 9-byte NOP + 2-byte NOP
        let mut enc = X86Encoder::new();
        enc.encode_multibyte_nop(11);
        let bytes = enc.finish();
        assert_eq!(bytes.len(), 11);
        // First 9 bytes: 66 0F 1F 84 00 00 00 00 00
        assert_eq!(&bytes[0..9], &[0x66, 0x0F, 0x1F, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00]);
        // Last 2 bytes: 66 90
        assert_eq!(&bytes[9..11], &[0x66, 0x90]);
    }

    #[test]
    fn test_nopmulti_via_instruction_default() {
        // NopMulti with imm=0 defaults to 3-byte NOP
        let bytes = encode(X86Opcode::NopMulti, &X86InstOperands::none());
        assert_eq!(bytes, vec![0x0F, 0x1F, 0x00]);
    }

    #[test]
    fn test_nopmulti_via_instruction_size_5() {
        // NopMulti with imm=5 produces 5-byte NOP
        let bytes = encode(
            X86Opcode::NopMulti,
            &X86InstOperands { imm: 5, ..X86InstOperands::none() },
        );
        assert_eq!(bytes, vec![0x0F, 0x1F, 0x44, 0x00, 0x00]);
    }

    #[test]
    fn test_nopmulti_via_instruction_size_8() {
        // NopMulti with imm=8 produces 8-byte NOP
        let bytes = encode(
            X86Opcode::NopMulti,
            &X86InstOperands { imm: 8, ..X86InstOperands::none() },
        );
        assert_eq!(bytes, vec![0x0F, 0x1F, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00]);
    }

    // -----------------------------------------------------------------------
    // New instruction size tests (wave 38)
    // -----------------------------------------------------------------------

    #[test]
    fn test_new_instruction_sizes_v3() {
        let mut enc = X86Encoder::new();

        // CDQ = 1 byte (99)
        let n = enc.encode_instruction(X86Opcode::Cdq, &X86InstOperands::none()).unwrap();
        assert_eq!(n, 1);

        // CQO = 2 bytes (REX.W + 99)
        let n = enc.encode_instruction(X86Opcode::Cqo, &X86InstOperands::none()).unwrap();
        assert_eq!(n, 2);

        // CMP r64, imm8 = 4 bytes (REX.W + 83 + ModRM + imm8)
        let n = enc.encode_instruction(X86Opcode::CmpRI8, &X86InstOperands::ri(RAX, 1)).unwrap();
        assert_eq!(n, 4);

        // CMP R15, imm8 = 4 bytes (REX.WB + 83 + ModRM + imm8)
        let n = enc.encode_instruction(X86Opcode::CmpRI8, &X86InstOperands::ri(R15, 1)).unwrap();
        assert_eq!(n, 4);

        // NopMulti default = 3 bytes (0F 1F 00)
        let n = enc.encode_instruction(X86Opcode::NopMulti, &X86InstOperands::none()).unwrap();
        assert_eq!(n, 3);

        // NopMulti size=9 = 9 bytes
        let n = enc.encode_instruction(
            X86Opcode::NopMulti,
            &X86InstOperands { imm: 9, ..X86InstOperands::none() },
        ).unwrap();
        assert_eq!(n, 9);
    }

    // -----------------------------------------------------------------------
    // All SETcc condition codes (comprehensive)
    // Intel SDM Vol 2: SETcc: 0F 90+cc /0
    // -----------------------------------------------------------------------

    #[test]
    fn test_setcc_all_conditions() {
        let all_cc = [
            (X86CondCode::O,  0x90u8),
            (X86CondCode::NO, 0x91),
            (X86CondCode::B,  0x92),
            (X86CondCode::AE, 0x93),
            (X86CondCode::E,  0x94),
            (X86CondCode::NE, 0x95),
            (X86CondCode::BE, 0x96),
            (X86CondCode::A,  0x97),
            (X86CondCode::S,  0x98),
            (X86CondCode::NS, 0x99),
            (X86CondCode::P,  0x9A),
            (X86CondCode::NP, 0x9B),
            (X86CondCode::L,  0x9C),
            (X86CondCode::GE, 0x9D),
            (X86CondCode::LE, 0x9E),
            (X86CondCode::G,  0x9F),
        ];
        for (cc, expected_byte) in &all_cc {
            let bytes = encode(
                X86Opcode::Setcc,
                &X86InstOperands {
                    dst: Some(AL),
                    cc: Some(*cc),
                    ..X86InstOperands::none()
                },
            );
            // 0F + cc_byte + ModRM(11 000 000)
            assert_eq!(bytes[0], 0x0F, "SETcc {:?} prefix", cc);
            assert_eq!(bytes[1], *expected_byte, "SETcc {:?} opcode byte", cc);
            assert_eq!(bytes[2], 0xC0, "SETcc {:?} ModRM", cc);
        }
    }

    // -----------------------------------------------------------------------
    // XCHG encoding (Intel SDM: 87 /r)
    // -----------------------------------------------------------------------

    #[test]
    fn test_xchg_rax_rcx() {
        // XCHG RAX, RCX: REX.W + 87 + ModRM(11_001_000)
        // encode_alu_rr puts src(RCX=1) in reg, dst(RAX=0) in rm
        // REX: W=1,R=0,X=0,B=0 = 0x48
        // ModRM: mod=11, reg=001, rm=000 = 0xC8
        let bytes = encode(X86Opcode::Xchg, &X86InstOperands::rr(RAX, RCX));
        assert_eq!(bytes, &[0x48, 0x87, 0xC8]);
    }

    #[test]
    fn test_xchg_r8_rdx() {
        // XCHG R8, RDX: src=RDX(2) in reg, dst=R8(8) in rm
        // REX: W=1,R=0,X=0,B=1(R8>=8) = 0x49
        // ModRM: mod=11, reg=010, rm=000(R8&7=0) = 0xD0
        let bytes = encode(X86Opcode::Xchg, &X86InstOperands::rr(R8, RDX));
        assert_eq!(bytes, &[0x49, 0x87, 0xD0]);
    }

    #[test]
    fn test_xchg_r15_r14() {
        // XCHG R15, R14: src=R14(14) in reg, dst=R15(15) in rm
        // REX: W=1,R=1(R14>=8),X=0,B=1(R15>=8) = 0x4D
        // ModRM: mod=11, reg=110(R14&7=6), rm=111(R15&7=7) = 0xF7
        let bytes = encode(X86Opcode::Xchg, &X86InstOperands::rr(R15, R14));
        assert_eq!(bytes, &[0x4D, 0x87, 0xF7]);
    }

    // -----------------------------------------------------------------------
    // CMPXCHG encoding (Intel SDM: F0 + 0F B1 /r)
    // -----------------------------------------------------------------------

    #[test]
    fn test_cmpxchg_rcx_rdx() {
        // LOCK CMPXCHG RCX, RDX
        // F0(LOCK) + REX.W(0x48) + 0F B1 + ModRM
        // src=RDX(2) in reg, dst=RCX(1) in rm
        // ModRM: mod=11, reg=010, rm=001 = 0xD1
        let bytes = encode(X86Opcode::Cmpxchg, &X86InstOperands::rr(RCX, RDX));
        assert_eq!(bytes, &[0xF0, 0x48, 0x0F, 0xB1, 0xD1]);
    }

    #[test]
    fn test_cmpxchg_r8_r9() {
        // LOCK CMPXCHG R8, R9
        // F0 + REX(W=1,R=1(R9>=8),B=1(R8>=8)) = REX 0x4D
        // 0F B1 + ModRM: mod=11, reg=001(R9&7=1), rm=000(R8&7=0) = 0xC8
        let bytes = encode(X86Opcode::Cmpxchg, &X86InstOperands::rr(R8, R9));
        assert_eq!(bytes, &[0xF0, 0x4D, 0x0F, 0xB1, 0xC8]);
    }

    // -----------------------------------------------------------------------
    // BT encoding (Intel SDM: 0F BA /4 ib)
    // -----------------------------------------------------------------------

    #[test]
    fn test_bt_rax_imm5() {
        // BT RAX, 5: REX.W(0x48) + 0F BA + ModRM(11_100_000) + 05
        // ModRM ext_reg(4, RAX=0): mod=11, reg=100, rm=000 = 0xE0
        let bytes = encode(X86Opcode::BtRI, &X86InstOperands::ri(RAX, 5));
        assert_eq!(bytes, &[0x48, 0x0F, 0xBA, 0xE0, 0x05]);
    }

    #[test]
    fn test_bt_r15_imm63() {
        // BT R15, 63: REX.WB(0x49) + 0F BA + ModRM(11_100_111) + 3F
        // ModRM ext_reg(4, R15&7=7): mod=11, reg=100, rm=111 = 0xE7
        let bytes = encode(X86Opcode::BtRI, &X86InstOperands::ri(R15, 63));
        assert_eq!(bytes, &[0x49, 0x0F, 0xBA, 0xE7, 0x3F]);
    }

    // -----------------------------------------------------------------------
    // BSWAP encoding (Intel SDM: 0F C8+rd)
    // -----------------------------------------------------------------------

    #[test]
    fn test_bswap_rax() {
        // BSWAP RAX: REX.W(0x48) + 0F + C8+0 = [0x48, 0x0F, 0xC8]
        let bytes = encode(X86Opcode::Bswap, &X86InstOperands::r(RAX));
        assert_eq!(bytes, &[0x48, 0x0F, 0xC8]);
    }

    #[test]
    fn test_bswap_rbx() {
        // BSWAP RBX: REX.W(0x48) + 0F + C8+3 = [0x48, 0x0F, 0xCB]
        let bytes = encode(X86Opcode::Bswap, &X86InstOperands::r(RBX));
        assert_eq!(bytes, &[0x48, 0x0F, 0xCB]);
    }

    #[test]
    fn test_bswap_r12() {
        // BSWAP R12: REX.WB(0x49) + 0F + C8+4(R12&7=4) = [0x49, 0x0F, 0xCC]
        let bytes = encode(X86Opcode::Bswap, &X86InstOperands::r(R12));
        assert_eq!(bytes, &[0x49, 0x0F, 0xCC]);
    }

    // -----------------------------------------------------------------------
    // MOVD/MOVQ xmm<->gpr encoding (Intel SDM: 66 0F 6E/7E)
    // -----------------------------------------------------------------------

    #[test]
    fn test_movd_to_xmm_xmm0_rax() {
        // MOVD XMM0, EAX: 66 0F 6E ModRM(11_000_000)=0xC0
        // No REX needed (both < 8)
        let bytes = encode(X86Opcode::MovdToXmm, &X86InstOperands::rr(XMM0, RAX));
        assert_eq!(bytes, &[0x66, 0x0F, 0x6E, 0xC0]);
    }

    #[test]
    fn test_movd_to_xmm8_rcx() {
        // MOVD XMM8, ECX: 66 REX.R(XMM8>=8)(0x44) 0F 6E ModRM(11_000_001)=0xC1
        // REX: W=0,R=1(XMM8>=8),X=0,B=0 = 0x44
        let bytes = encode(X86Opcode::MovdToXmm, &X86InstOperands::rr(XMM8, RCX));
        assert_eq!(bytes, &[0x66, 0x44, 0x0F, 0x6E, 0xC1]);
    }

    #[test]
    fn test_movd_from_xmm_rax_xmm0() {
        // MOVD EAX, XMM0: 66 0F 7E ModRM(11_000_000)=0xC0
        // src=XMM0 in reg, dst=RAX in rm
        let bytes = encode(X86Opcode::MovdFromXmm, &X86InstOperands::rr(RAX, XMM0));
        assert_eq!(bytes, &[0x66, 0x0F, 0x7E, 0xC0]);
    }

    #[test]
    fn test_movq_to_xmm_xmm0_rax() {
        // MOVQ XMM0, RAX: 66 REX.W(0x48) 0F 6E ModRM(11_000_000)=0xC0
        let bytes = encode(X86Opcode::MovqToXmm, &X86InstOperands::rr(XMM0, RAX));
        assert_eq!(bytes, &[0x66, 0x48, 0x0F, 0x6E, 0xC0]);
    }

    #[test]
    fn test_movq_from_xmm_rax_xmm0() {
        // MOVQ RAX, XMM0: 66 REX.W(0x48) 0F 7E ModRM(11_000_000)=0xC0
        let bytes = encode(X86Opcode::MovqFromXmm, &X86InstOperands::rr(RAX, XMM0));
        assert_eq!(bytes, &[0x66, 0x48, 0x0F, 0x7E, 0xC0]);
    }

    #[test]
    fn test_movq_to_xmm15_r15() {
        // MOVQ XMM15, R15: 66 REX.WRB(0x4D) 0F 6E ModRM(11_111_111)=0xFF
        // REX: W=1, R=1(XMM15>=8), X=0, B=1(R15>=8) = 0x4D
        // ModRM: mod=11, reg=111(XMM15&7=7), rm=111(R15&7=7) = 0xFF
        let bytes = encode(X86Opcode::MovqToXmm, &X86InstOperands::rr(XMM15, R15));
        assert_eq!(bytes, &[0x66, 0x4D, 0x0F, 0x6E, 0xFF]);
    }

    #[test]
    fn test_movq_from_xmm15_r15() {
        // MOVQ R15, XMM15: 66 REX.WRB(0x4D) 0F 7E ModRM(11_111_111)=0xFF
        // src=XMM15 in reg, dst=R15 in rm
        // REX: W=1, R=1(XMM15>=8), X=0, B=1(R15>=8) = 0x4D
        let bytes = encode(X86Opcode::MovqFromXmm, &X86InstOperands::rr(R15, XMM15));
        assert_eq!(bytes, &[0x66, 0x4D, 0x0F, 0x7E, 0xFF]);
    }

    // -----------------------------------------------------------------------
    // Instruction size sanity checks for new opcodes
    // -----------------------------------------------------------------------

    #[test]
    fn test_new_instruction_sizes_v4() {
        let mut enc = X86Encoder::new();

        // XCHG RAX,RCX = 3 bytes (REX.W + 87 + ModRM)
        let n = enc.encode_instruction(X86Opcode::Xchg, &X86InstOperands::rr(RAX, RCX)).unwrap();
        assert_eq!(n, 3, "XCHG RAX,RCX size");

        // CMPXCHG RCX,RDX = 5 bytes (LOCK + REX.W + 0F + B1 + ModRM)
        let n = enc.encode_instruction(X86Opcode::Cmpxchg, &X86InstOperands::rr(RCX, RDX)).unwrap();
        assert_eq!(n, 5, "CMPXCHG RCX,RDX size");

        // BT RAX,5 = 5 bytes (REX.W + 0F + BA + ModRM + imm8)
        let n = enc.encode_instruction(X86Opcode::BtRI, &X86InstOperands::ri(RAX, 5)).unwrap();
        assert_eq!(n, 5, "BT RAX,5 size");

        // BSWAP RAX = 3 bytes (REX.W + 0F + C8)
        let n = enc.encode_instruction(X86Opcode::Bswap, &X86InstOperands::r(RAX)).unwrap();
        assert_eq!(n, 3, "BSWAP RAX size");

        // MOVD XMM0,EAX = 4 bytes (66 + 0F + 6E + ModRM)
        let n = enc.encode_instruction(X86Opcode::MovdToXmm, &X86InstOperands::rr(XMM0, RAX)).unwrap();
        assert_eq!(n, 4, "MOVD XMM0,EAX size");

        // MOVQ XMM0,RAX = 5 bytes (66 + REX.W + 0F + 6E + ModRM)
        let n = enc.encode_instruction(X86Opcode::MovqToXmm, &X86InstOperands::rr(XMM0, RAX)).unwrap();
        assert_eq!(n, 5, "MOVQ XMM0,RAX size");
    }

    // -----------------------------------------------------------------------
    // RIP-relative SSE load tests (MOVSS/MOVSD [RIP+disp32])
    //
    // Intel SDM Vol 2:
    //   MOVSS xmm, m32: F3 0F 10 /r
    //   MOVSD xmm, m64: F2 0F 10 /r
    // RIP-relative: ModRM mod=00, rm=101 signals [RIP+disp32] in 64-bit mode
    // REX.R needed for XMM8-XMM15 (extends ModRM reg field)
    // -----------------------------------------------------------------------

    #[test]
    fn test_movss_rip_rel_xmm0_disp0() {
        // MOVSS XMM0, [RIP+0]: F3 0F 10 ModRM(00 000 101) disp32(00000000)
        // No REX needed (XMM0 hw_enc=0)
        let bytes = encode(X86Opcode::MovssRipRel, &X86InstOperands::rip_rel(XMM0, 0));
        assert_eq!(bytes, vec![0xF3, 0x0F, 0x10, 0x05, 0x00, 0x00, 0x00, 0x00]);
    }

    #[test]
    fn test_movsd_rip_rel_xmm0_disp0() {
        // MOVSD XMM0, [RIP+0]: F2 0F 10 ModRM(00 000 101) disp32(00000000)
        // No REX needed (XMM0 hw_enc=0)
        let bytes = encode(X86Opcode::MovsdRipRel, &X86InstOperands::rip_rel(XMM0, 0));
        assert_eq!(bytes, vec![0xF2, 0x0F, 0x10, 0x05, 0x00, 0x00, 0x00, 0x00]);
    }

    #[test]
    fn test_movsd_rip_rel_xmm1_disp256() {
        // MOVSD XMM1, [RIP+256]: F2 0F 10 ModRM(00 001 101) disp32
        // XMM1 reg=001 -> ModRM = 00_001_101 = 0x0D
        let bytes = encode(X86Opcode::MovsdRipRel, &X86InstOperands::rip_rel(XMM1, 256));
        assert_eq!(bytes, vec![0xF2, 0x0F, 0x10, 0x0D, 0x00, 0x01, 0x00, 0x00]);
    }

    #[test]
    fn test_movss_rip_rel_xmm8_disp0() {
        // MOVSS XMM8, [RIP+0]: F3 REX.R(44) 0F 10 ModRM(00 000 101) disp32
        // XMM8 hw_enc=8, needs REX.R: 0100 0100 = 0x44
        let bytes = encode(X86Opcode::MovssRipRel, &X86InstOperands::rip_rel(XMM8, 0));
        assert_eq!(bytes, vec![0xF3, 0x44, 0x0F, 0x10, 0x05, 0x00, 0x00, 0x00, 0x00]);
    }

    #[test]
    fn test_movsd_rip_rel_xmm8_negative_disp() {
        // MOVSD XMM8, [RIP-16]: F2 REX.R(44) 0F 10 ModRM(00 000 101) disp32(F0FFFFFF)
        let bytes = encode(X86Opcode::MovsdRipRel, &X86InstOperands::rip_rel(XMM8, -16));
        assert_eq!(bytes, vec![0xF2, 0x44, 0x0F, 0x10, 0x05, 0xF0, 0xFF, 0xFF, 0xFF]);
    }

    #[test]
    fn test_movss_rip_rel_xmm15_disp128() {
        // MOVSS XMM15, [RIP+128]: F3 REX.R(44) 0F 10 ModRM(00 111 101) disp32
        // XMM15 hw_enc=15, reg&7=7 -> ModRM = 00_111_101 = 0x3D
        let bytes = encode(X86Opcode::MovssRipRel, &X86InstOperands::rip_rel(XMM15, 128));
        assert_eq!(bytes, vec![0xF3, 0x44, 0x0F, 0x10, 0x3D, 0x80, 0x00, 0x00, 0x00]);
    }

    #[test]
    fn test_movss_rip_rel_size_no_rex() {
        // MOVSS XMM0, [RIP+0] should be 8 bytes (no REX)
        // F3(1) + 0F(1) + 10(1) + ModRM(1) + disp32(4) = 8
        let mut enc = X86Encoder::new();
        let n = enc.encode_instruction(
            X86Opcode::MovssRipRel,
            &X86InstOperands::rip_rel(XMM0, 0),
        ).unwrap();
        assert_eq!(n, 8, "MOVSS XMM0,[RIP+0] size without REX");
    }

    #[test]
    fn test_movsd_rip_rel_size_with_rex() {
        // MOVSD XMM8, [RIP+0] should be 9 bytes (with REX.R)
        // F2(1) + REX(1) + 0F(1) + 10(1) + ModRM(1) + disp32(4) = 9
        let mut enc = X86Encoder::new();
        let n = enc.encode_instruction(
            X86Opcode::MovsdRipRel,
            &X86InstOperands::rip_rel(XMM8, 0),
        ).unwrap();
        assert_eq!(n, 9, "MOVSD XMM8,[RIP+0] size with REX.R");
    }

    #[test]
    fn test_movss_rip_rel_missing_dst_error() {
        let mut enc = X86Encoder::new();
        let result = enc.encode_instruction(X86Opcode::MovssRipRel, &X86InstOperands::none());
        assert!(result.is_err(), "MovssRipRel without dst should fail");
    }

    #[test]
    fn test_movsd_rip_rel_missing_dst_error() {
        let mut enc = X86Encoder::new();
        let result = enc.encode_instruction(X86Opcode::MovsdRipRel, &X86InstOperands::none());
        assert!(result.is_err(), "MovsdRipRel without dst should fail");
    }
}
