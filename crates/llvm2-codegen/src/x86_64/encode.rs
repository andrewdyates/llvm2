// llvm2-codegen/x86_64/encode.rs - x86-64 instruction binary encoder
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
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
}
