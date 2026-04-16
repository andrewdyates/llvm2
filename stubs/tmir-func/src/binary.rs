// tmir-func/binary.rs — Binary tMIR bitcode encoder/decoder (.tmbc)
//
// Author: Andrew Yates <ayates@dropbox.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0
//
// Implements Layer 2 of the tMIR transport architecture: a custom binary format
// inspired by WASM binary and LLVM bitcode. Uses LEB128 for integers, section-based
// structure, deterministic/canonical encoding.
//
// Format:
//   Magic: b"tMBC" (4 bytes)
//   Version: u32-leb128
//   Sections: [section_id: u8, payload_size: u32-leb128, payload: bytes]*
//
// Section IDs:
//   1 = Type (deduplicated type table)
//   2 = Struct (struct definitions)
//   3 = Global (global variable definitions)
//   4 = Function (function headers: id, name, signature, entry, proofs)
//   5 = Code (function bodies: blocks -> instructions)
//   6 = DataLayout (target data layout)
//
// Reference: designs/2026-04-16-tmir-transport-architecture.md

use crate::Module;
use std::error::Error;
use std::fmt;

// ---------------------------------------------------------------------------
// Magic and version
// ---------------------------------------------------------------------------

const MAGIC: &[u8; 4] = b"tMBC";
const VERSION: u32 = 1;

// ---------------------------------------------------------------------------
// Section IDs
// ---------------------------------------------------------------------------

const SECTION_TYPE: u8 = 1;
const SECTION_STRUCT: u8 = 2;
const SECTION_GLOBAL: u8 = 3;
const SECTION_FUNCTION: u8 = 4;
const SECTION_CODE: u8 = 5;
const SECTION_DATA_LAYOUT: u8 = 6;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Error type for tMIR binary read/write operations.
#[derive(Debug)]
pub enum TmirBinaryError {
    /// Unexpected end of input.
    UnexpectedEof,
    /// Invalid magic bytes.
    InvalidMagic([u8; 4]),
    /// Unsupported version.
    UnsupportedVersion(u32),
    /// Unknown section ID.
    UnknownSection(u8),
    /// Unknown type tag.
    UnknownTypeTag(u8),
    /// Unknown instruction opcode.
    UnknownOpcode(u8),
    /// Unknown enum variant tag.
    UnknownTag(&'static str, u8),
    /// LEB128 overflow (value too large).
    Leb128Overflow,
    /// UTF-8 decoding error.
    InvalidUtf8,
    /// Structural validation error.
    Validation(String),
    /// I/O error (file read/write).
    IoError(std::io::Error),
}

impl fmt::Display for TmirBinaryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnexpectedEof => write!(f, "unexpected end of input"),
            Self::InvalidMagic(m) => write!(f, "invalid magic: {:?}", m),
            Self::UnsupportedVersion(v) => write!(f, "unsupported version: {v}"),
            Self::UnknownSection(id) => write!(f, "unknown section id: {id}"),
            Self::UnknownTypeTag(t) => write!(f, "unknown type tag: {t}"),
            Self::UnknownOpcode(op) => write!(f, "unknown instruction opcode: {op}"),
            Self::UnknownTag(ctx, t) => write!(f, "unknown {ctx} tag: {t}"),
            Self::Leb128Overflow => write!(f, "LEB128 value overflow"),
            Self::InvalidUtf8 => write!(f, "invalid UTF-8 string"),
            Self::Validation(msg) => write!(f, "validation: {msg}"),
            Self::IoError(err) => write!(f, "I/O error: {err}"),
        }
    }
}

impl Error for TmirBinaryError {}

// ---------------------------------------------------------------------------
// LEB128 encoder/decoder
// ---------------------------------------------------------------------------

/// Encode an unsigned integer as LEB128.
fn encode_uleb128(buf: &mut Vec<u8>, mut value: u64) {
    loop {
        let mut byte = (value & 0x7F) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
        }
        buf.push(byte);
        if value == 0 {
            break;
        }
    }
}

/// Encode a signed integer as SLEB128.
fn encode_sleb128(buf: &mut Vec<u8>, mut value: i128) {
    loop {
        let byte = (value & 0x7F) as u8;
        value >>= 7;
        let sign_bit = (byte & 0x40) != 0;
        if (value == 0 && !sign_bit) || (value == -1 && sign_bit) {
            buf.push(byte);
            break;
        }
        buf.push(byte | 0x80);
    }
}

/// Decode an unsigned LEB128 integer.
fn decode_uleb128(data: &[u8], pos: &mut usize) -> Result<u64, TmirBinaryError> {
    let mut result: u64 = 0;
    let mut shift: u32 = 0;
    loop {
        if *pos >= data.len() {
            return Err(TmirBinaryError::UnexpectedEof);
        }
        let byte = data[*pos];
        *pos += 1;
        let payload = (byte & 0x7F) as u64;
        if shift >= 64 || (shift == 63 && payload > 1) {
            return Err(TmirBinaryError::Leb128Overflow);
        }
        result |= payload << shift;
        if byte & 0x80 == 0 {
            return Ok(result);
        }
        shift += 7;
    }
}

/// Decode a signed SLEB128 integer.
fn decode_sleb128(data: &[u8], pos: &mut usize) -> Result<i128, TmirBinaryError> {
    let mut result: i128 = 0;
    let mut shift: u32 = 0;
    let mut byte;
    loop {
        if *pos >= data.len() {
            return Err(TmirBinaryError::UnexpectedEof);
        }
        byte = data[*pos];
        *pos += 1;
        result |= ((byte & 0x7F) as i128) << shift;
        shift += 7;
        if byte & 0x80 == 0 {
            break;
        }
        if shift >= 128 {
            return Err(TmirBinaryError::Leb128Overflow);
        }
    }
    // Sign extend
    if shift < 128 && (byte & 0x40) != 0 {
        result |= !0i128 << shift;
    }
    Ok(result)
}

// ---------------------------------------------------------------------------
// Encoder helpers
// ---------------------------------------------------------------------------

fn encode_u8(buf: &mut Vec<u8>, v: u8) {
    buf.push(v);
}

fn encode_u32(buf: &mut Vec<u8>, v: u32) {
    encode_uleb128(buf, v as u64);
}

fn encode_u64(buf: &mut Vec<u8>, v: u64) {
    encode_uleb128(buf, v);
}

fn encode_i32(buf: &mut Vec<u8>, v: i32) {
    encode_sleb128(buf, v as i128);
}

fn encode_i64(buf: &mut Vec<u8>, v: i64) {
    encode_sleb128(buf, v as i128);
}

fn encode_i128(buf: &mut Vec<u8>, v: i128) {
    encode_sleb128(buf, v);
}

fn encode_f64(buf: &mut Vec<u8>, v: f64) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn encode_bool(buf: &mut Vec<u8>, v: bool) {
    buf.push(if v { 1 } else { 0 });
}

fn encode_string(buf: &mut Vec<u8>, s: &str) {
    encode_u32(buf, s.len() as u32);
    buf.extend_from_slice(s.as_bytes());
}

fn encode_bytes(buf: &mut Vec<u8>, b: &[u8]) {
    encode_u32(buf, b.len() as u32);
    buf.extend_from_slice(b);
}

// ---------------------------------------------------------------------------
// Decoder helpers
// ---------------------------------------------------------------------------

fn decode_u8(data: &[u8], pos: &mut usize) -> Result<u8, TmirBinaryError> {
    if *pos >= data.len() {
        return Err(TmirBinaryError::UnexpectedEof);
    }
    let v = data[*pos];
    *pos += 1;
    Ok(v)
}

fn decode_u32(data: &[u8], pos: &mut usize) -> Result<u32, TmirBinaryError> {
    let v = decode_uleb128(data, pos)?;
    if v > u32::MAX as u64 {
        return Err(TmirBinaryError::Leb128Overflow);
    }
    Ok(v as u32)
}

fn decode_u64(data: &[u8], pos: &mut usize) -> Result<u64, TmirBinaryError> {
    decode_uleb128(data, pos)
}

fn decode_i32(data: &[u8], pos: &mut usize) -> Result<i32, TmirBinaryError> {
    let v = decode_sleb128(data, pos)?;
    if v < i32::MIN as i128 || v > i32::MAX as i128 {
        return Err(TmirBinaryError::Leb128Overflow);
    }
    Ok(v as i32)
}

fn decode_i64(data: &[u8], pos: &mut usize) -> Result<i64, TmirBinaryError> {
    let v = decode_sleb128(data, pos)?;
    if v < i64::MIN as i128 || v > i64::MAX as i128 {
        return Err(TmirBinaryError::Leb128Overflow);
    }
    Ok(v as i64)
}

fn decode_i128(data: &[u8], pos: &mut usize) -> Result<i128, TmirBinaryError> {
    decode_sleb128(data, pos)
}

fn decode_f64(data: &[u8], pos: &mut usize) -> Result<f64, TmirBinaryError> {
    if *pos + 8 > data.len() {
        return Err(TmirBinaryError::UnexpectedEof);
    }
    let bytes: [u8; 8] = data[*pos..*pos + 8].try_into().unwrap();
    *pos += 8;
    Ok(f64::from_le_bytes(bytes))
}

fn decode_bool(data: &[u8], pos: &mut usize) -> Result<bool, TmirBinaryError> {
    let v = decode_u8(data, pos)?;
    Ok(v != 0)
}

fn decode_string(data: &[u8], pos: &mut usize) -> Result<String, TmirBinaryError> {
    let len = decode_u32(data, pos)? as usize;
    if *pos + len > data.len() {
        return Err(TmirBinaryError::UnexpectedEof);
    }
    let s = std::str::from_utf8(&data[*pos..*pos + len])
        .map_err(|_| TmirBinaryError::InvalidUtf8)?;
    *pos += len;
    Ok(s.to_string())
}

fn decode_byte_vec(data: &[u8], pos: &mut usize) -> Result<Vec<u8>, TmirBinaryError> {
    let len = decode_u32(data, pos)? as usize;
    if *pos + len > data.len() {
        return Err(TmirBinaryError::UnexpectedEof);
    }
    let v = data[*pos..*pos + len].to_vec();
    *pos += len;
    Ok(v)
}

// ---------------------------------------------------------------------------
// Type encoding/decoding
// ---------------------------------------------------------------------------

// Ty variant tags (deterministic, canonical)
const TY_PRIMITIVE_INT: u8 = 0;
const TY_PRIMITIVE_FLOAT: u8 = 1;
const TY_PRIMITIVE_BOOL: u8 = 2;
const TY_PRIMITIVE_UNIT: u8 = 3;
const TY_PRIMITIVE_NEVER: u8 = 4;
const TY_ARRAY: u8 = 5;
const TY_TUPLE: u8 = 6;
const TY_STRUCT_DEF: u8 = 7;
const TY_ENUM: u8 = 8;
const TY_REF: u8 = 9;
const TY_VECTOR: u8 = 10;
const TY_FNPTR: u8 = 11;
const TY_STRUCT_ID: u8 = 12;
const TY_FUNC: u8 = 13;

use tmir_types::*;

fn encode_int_width(buf: &mut Vec<u8>, w: IntWidth) {
    let tag: u8 = match w {
        IntWidth::I8 => 0,
        IntWidth::I16 => 1,
        IntWidth::I32 => 2,
        IntWidth::I64 => 3,
        IntWidth::I128 => 4,
    };
    encode_u8(buf, tag);
}

fn decode_int_width(data: &[u8], pos: &mut usize) -> Result<IntWidth, TmirBinaryError> {
    let tag = decode_u8(data, pos)?;
    match tag {
        0 => Ok(IntWidth::I8),
        1 => Ok(IntWidth::I16),
        2 => Ok(IntWidth::I32),
        3 => Ok(IntWidth::I64),
        4 => Ok(IntWidth::I128),
        _ => Err(TmirBinaryError::UnknownTag("IntWidth", tag)),
    }
}

fn encode_float_width(buf: &mut Vec<u8>, w: FloatWidth) {
    let tag: u8 = match w {
        FloatWidth::F32 => 0,
        FloatWidth::F64 => 1,
    };
    encode_u8(buf, tag);
}

fn decode_float_width(data: &[u8], pos: &mut usize) -> Result<FloatWidth, TmirBinaryError> {
    let tag = decode_u8(data, pos)?;
    match tag {
        0 => Ok(FloatWidth::F32),
        1 => Ok(FloatWidth::F64),
        _ => Err(TmirBinaryError::UnknownTag("FloatWidth", tag)),
    }
}

fn encode_mutability(buf: &mut Vec<u8>, m: Mutability) {
    encode_u8(buf, match m { Mutability::Immutable => 0, Mutability::Mutable => 1 });
}

fn decode_mutability(data: &[u8], pos: &mut usize) -> Result<Mutability, TmirBinaryError> {
    let tag = decode_u8(data, pos)?;
    match tag {
        0 => Ok(Mutability::Immutable),
        1 => Ok(Mutability::Mutable),
        _ => Err(TmirBinaryError::UnknownTag("Mutability", tag)),
    }
}

fn encode_ref_kind(buf: &mut Vec<u8>, kind: &RefKind) {
    match kind {
        RefKind::Borrow(m) => { encode_u8(buf, 0); encode_mutability(buf, *m); }
        RefKind::Raw(m) => { encode_u8(buf, 1); encode_mutability(buf, *m); }
        RefKind::Rc => { encode_u8(buf, 2); }
    }
}

fn decode_ref_kind(data: &[u8], pos: &mut usize) -> Result<RefKind, TmirBinaryError> {
    let tag = decode_u8(data, pos)?;
    match tag {
        0 => Ok(RefKind::Borrow(decode_mutability(data, pos)?)),
        1 => Ok(RefKind::Raw(decode_mutability(data, pos)?)),
        2 => Ok(RefKind::Rc),
        _ => Err(TmirBinaryError::UnknownTag("RefKind", tag)),
    }
}

fn encode_linkage(buf: &mut Vec<u8>, l: Linkage) {
    encode_u8(buf, match l {
        Linkage::External => 0,
        Linkage::Internal => 1,
        Linkage::Weak => 2,
        Linkage::AvailableExternally => 3,
    });
}

fn decode_linkage(data: &[u8], pos: &mut usize) -> Result<Linkage, TmirBinaryError> {
    let tag = decode_u8(data, pos)?;
    match tag {
        0 => Ok(Linkage::External),
        1 => Ok(Linkage::Internal),
        2 => Ok(Linkage::Weak),
        3 => Ok(Linkage::AvailableExternally),
        _ => Err(TmirBinaryError::UnknownTag("Linkage", tag)),
    }
}

fn encode_ty(buf: &mut Vec<u8>, ty: &Ty) {
    match ty {
        Ty::Primitive(PrimitiveType::Int { width, signed }) => {
            encode_u8(buf, TY_PRIMITIVE_INT);
            encode_int_width(buf, *width);
            encode_bool(buf, *signed);
        }
        Ty::Primitive(PrimitiveType::Float(fw)) => {
            encode_u8(buf, TY_PRIMITIVE_FLOAT);
            encode_float_width(buf, *fw);
        }
        Ty::Primitive(PrimitiveType::Bool) => encode_u8(buf, TY_PRIMITIVE_BOOL),
        Ty::Primitive(PrimitiveType::Unit) => encode_u8(buf, TY_PRIMITIVE_UNIT),
        Ty::Primitive(PrimitiveType::Never) => encode_u8(buf, TY_PRIMITIVE_NEVER),
        Ty::Array { element, len } => {
            encode_u8(buf, TY_ARRAY);
            encode_ty(buf, element);
            encode_u64(buf, *len);
        }
        Ty::Tuple(elems) => {
            encode_u8(buf, TY_TUPLE);
            encode_u32(buf, elems.len() as u32);
            for e in elems {
                encode_ty(buf, e);
            }
        }
        Ty::StructDef { name, fields } => {
            encode_u8(buf, TY_STRUCT_DEF);
            encode_string(buf, name);
            encode_u32(buf, fields.len() as u32);
            for f in fields {
                encode_string(buf, &f.name);
                encode_ty(buf, &f.ty);
            }
        }
        Ty::Enum { name, variants } => {
            encode_u8(buf, TY_ENUM);
            encode_string(buf, name);
            encode_u32(buf, variants.len() as u32);
            for v in variants {
                encode_string(buf, &v.name);
                encode_u32(buf, v.fields.len() as u32);
                for f in &v.fields {
                    encode_ty(buf, f);
                }
            }
        }
        Ty::Ref { kind, pointee } => {
            encode_u8(buf, TY_REF);
            encode_ref_kind(buf, kind);
            encode_ty(buf, pointee);
        }
        Ty::Vector { element, lanes } => {
            encode_u8(buf, TY_VECTOR);
            encode_ty(buf, element);
            encode_u32(buf, *lanes);
        }
        Ty::FnPtr { params, ret } => {
            encode_u8(buf, TY_FNPTR);
            encode_u32(buf, params.len() as u32);
            for p in params {
                encode_ty(buf, p);
            }
            encode_ty(buf, ret);
        }
        Ty::Struct(StructId(id)) => {
            encode_u8(buf, TY_STRUCT_ID);
            encode_u32(buf, *id);
        }
        Ty::Func(FuncTy { params, returns }) => {
            encode_u8(buf, TY_FUNC);
            encode_u32(buf, params.len() as u32);
            for p in params {
                encode_ty(buf, p);
            }
            encode_u32(buf, returns.len() as u32);
            for r in returns {
                encode_ty(buf, r);
            }
        }
    }
}

fn decode_ty(data: &[u8], pos: &mut usize) -> Result<Ty, TmirBinaryError> {
    let tag = decode_u8(data, pos)?;
    match tag {
        TY_PRIMITIVE_INT => {
            let width = decode_int_width(data, pos)?;
            let signed = decode_bool(data, pos)?;
            Ok(Ty::Primitive(PrimitiveType::Int { width, signed }))
        }
        TY_PRIMITIVE_FLOAT => {
            let fw = decode_float_width(data, pos)?;
            Ok(Ty::Primitive(PrimitiveType::Float(fw)))
        }
        TY_PRIMITIVE_BOOL => Ok(Ty::Primitive(PrimitiveType::Bool)),
        TY_PRIMITIVE_UNIT => Ok(Ty::Primitive(PrimitiveType::Unit)),
        TY_PRIMITIVE_NEVER => Ok(Ty::Primitive(PrimitiveType::Never)),
        TY_ARRAY => {
            let element = decode_ty(data, pos)?;
            let len = decode_u64(data, pos)?;
            Ok(Ty::Array { element: Box::new(element), len })
        }
        TY_TUPLE => {
            let count = decode_u32(data, pos)? as usize;
            let mut elems = Vec::with_capacity(count);
            for _ in 0..count {
                elems.push(decode_ty(data, pos)?);
            }
            Ok(Ty::Tuple(elems))
        }
        TY_STRUCT_DEF => {
            let name = decode_string(data, pos)?;
            let count = decode_u32(data, pos)? as usize;
            let mut fields = Vec::with_capacity(count);
            for _ in 0..count {
                let fname = decode_string(data, pos)?;
                let fty = decode_ty(data, pos)?;
                fields.push(Field { name: fname, ty: fty });
            }
            Ok(Ty::StructDef { name, fields })
        }
        TY_ENUM => {
            let name = decode_string(data, pos)?;
            let count = decode_u32(data, pos)? as usize;
            let mut variants = Vec::with_capacity(count);
            for _ in 0..count {
                let vname = decode_string(data, pos)?;
                let fcount = decode_u32(data, pos)? as usize;
                let mut fields = Vec::with_capacity(fcount);
                for _ in 0..fcount {
                    fields.push(decode_ty(data, pos)?);
                }
                variants.push(Variant { name: vname, fields });
            }
            Ok(Ty::Enum { name, variants })
        }
        TY_REF => {
            let kind = decode_ref_kind(data, pos)?;
            let pointee = decode_ty(data, pos)?;
            Ok(Ty::Ref { kind, pointee: Box::new(pointee) })
        }
        TY_VECTOR => {
            let element = decode_ty(data, pos)?;
            let lanes = decode_u32(data, pos)?;
            Ok(Ty::Vector { element: Box::new(element), lanes })
        }
        TY_FNPTR => {
            let pcount = decode_u32(data, pos)? as usize;
            let mut params = Vec::with_capacity(pcount);
            for _ in 0..pcount {
                params.push(decode_ty(data, pos)?);
            }
            let ret = decode_ty(data, pos)?;
            Ok(Ty::FnPtr { params, ret: Box::new(ret) })
        }
        TY_STRUCT_ID => {
            let id = decode_u32(data, pos)?;
            Ok(Ty::Struct(StructId(id)))
        }
        TY_FUNC => {
            let pcount = decode_u32(data, pos)? as usize;
            let mut params = Vec::with_capacity(pcount);
            for _ in 0..pcount {
                params.push(decode_ty(data, pos)?);
            }
            let rcount = decode_u32(data, pos)? as usize;
            let mut returns = Vec::with_capacity(rcount);
            for _ in 0..rcount {
                returns.push(decode_ty(data, pos)?);
            }
            Ok(Ty::Func(FuncTy { params, returns }))
        }
        _ => Err(TmirBinaryError::UnknownTypeTag(tag)),
    }
}

// ---------------------------------------------------------------------------
// FuncTy encoding/decoding
// ---------------------------------------------------------------------------

fn encode_func_ty(buf: &mut Vec<u8>, ft: &FuncTy) {
    encode_u32(buf, ft.params.len() as u32);
    for p in &ft.params {
        encode_ty(buf, p);
    }
    encode_u32(buf, ft.returns.len() as u32);
    for r in &ft.returns {
        encode_ty(buf, r);
    }
}

fn decode_func_ty(data: &[u8], pos: &mut usize) -> Result<FuncTy, TmirBinaryError> {
    let pcount = decode_u32(data, pos)? as usize;
    let mut params = Vec::with_capacity(pcount);
    for _ in 0..pcount {
        params.push(decode_ty(data, pos)?);
    }
    let rcount = decode_u32(data, pos)? as usize;
    let mut returns = Vec::with_capacity(rcount);
    for _ in 0..rcount {
        returns.push(decode_ty(data, pos)?);
    }
    Ok(FuncTy { params, returns })
}

// ---------------------------------------------------------------------------
// TmirProof encoding/decoding
// ---------------------------------------------------------------------------

fn encode_proof(buf: &mut Vec<u8>, proof: &TmirProof) {
    match proof {
        TmirProof::Pure => encode_u8(buf, 0),
        TmirProof::ValidBorrow { borrow } => {
            encode_u8(buf, 1);
            encode_u32(buf, borrow.0);
        }
        TmirProof::InBounds { base, index } => {
            encode_u8(buf, 2);
            encode_u32(buf, base.0);
            encode_u32(buf, index.0);
        }
        TmirProof::Associative => encode_u8(buf, 3),
        TmirProof::Commutative => encode_u8(buf, 4),
        TmirProof::NoOverflow { signed } => {
            encode_u8(buf, 5);
            encode_bool(buf, *signed);
        }
        TmirProof::NotNull { ptr } => {
            encode_u8(buf, 6);
            encode_u32(buf, ptr.0);
        }
        TmirProof::NonZeroDivisor { divisor } => {
            encode_u8(buf, 7);
            encode_u32(buf, divisor.0);
        }
        TmirProof::ValidShift { amount, bitwidth } => {
            encode_u8(buf, 8);
            encode_u32(buf, amount.0);
            encode_u32(buf, *bitwidth as u32);
        }
        TmirProof::InRange { lo, hi } => {
            encode_u8(buf, 9);
            encode_i128(buf, *lo);
            encode_i128(buf, *hi);
        }
        TmirProof::Idempotent => encode_u8(buf, 10),
    }
}

fn decode_proof(data: &[u8], pos: &mut usize) -> Result<TmirProof, TmirBinaryError> {
    let tag = decode_u8(data, pos)?;
    match tag {
        0 => Ok(TmirProof::Pure),
        1 => {
            let borrow = ValueId(decode_u32(data, pos)?);
            Ok(TmirProof::ValidBorrow { borrow })
        }
        2 => {
            let base = ValueId(decode_u32(data, pos)?);
            let index = ValueId(decode_u32(data, pos)?);
            Ok(TmirProof::InBounds { base, index })
        }
        3 => Ok(TmirProof::Associative),
        4 => Ok(TmirProof::Commutative),
        5 => {
            let signed = decode_bool(data, pos)?;
            Ok(TmirProof::NoOverflow { signed })
        }
        6 => {
            let ptr = ValueId(decode_u32(data, pos)?);
            Ok(TmirProof::NotNull { ptr })
        }
        7 => {
            let divisor = ValueId(decode_u32(data, pos)?);
            Ok(TmirProof::NonZeroDivisor { divisor })
        }
        8 => {
            let amount = ValueId(decode_u32(data, pos)?);
            let bitwidth = decode_u32(data, pos)? as u16;
            Ok(TmirProof::ValidShift { amount, bitwidth })
        }
        9 => {
            let lo = decode_i128(data, pos)?;
            let hi = decode_i128(data, pos)?;
            Ok(TmirProof::InRange { lo, hi })
        }
        10 => Ok(TmirProof::Idempotent),
        _ => Err(TmirBinaryError::UnknownTag("TmirProof", tag)),
    }
}

fn encode_proofs(buf: &mut Vec<u8>, proofs: &[TmirProof]) {
    encode_u32(buf, proofs.len() as u32);
    for p in proofs {
        encode_proof(buf, p);
    }
}

fn decode_proofs(data: &[u8], pos: &mut usize) -> Result<Vec<TmirProof>, TmirBinaryError> {
    let count = decode_u32(data, pos)? as usize;
    let mut result = Vec::with_capacity(count);
    for _ in 0..count {
        result.push(decode_proof(data, pos)?);
    }
    Ok(result)
}

// ---------------------------------------------------------------------------
// Operand encoding/decoding
// ---------------------------------------------------------------------------

use tmir_instrs::{Constant, Operand};

fn encode_operand(buf: &mut Vec<u8>, op: &Operand) {
    match op {
        Operand::Value(vid) => {
            encode_u8(buf, 0);
            encode_u32(buf, vid.0);
        }
        Operand::Constant(c) => {
            encode_u8(buf, 1);
            encode_constant(buf, c);
        }
    }
}

fn decode_operand(data: &[u8], pos: &mut usize) -> Result<Operand, TmirBinaryError> {
    let tag = decode_u8(data, pos)?;
    match tag {
        0 => {
            let vid = ValueId(decode_u32(data, pos)?);
            Ok(Operand::Value(vid))
        }
        1 => {
            let c = decode_constant(data, pos)?;
            Ok(Operand::Constant(c))
        }
        _ => Err(TmirBinaryError::UnknownTag("Operand", tag)),
    }
}

fn encode_operands(buf: &mut Vec<u8>, ops: &[Operand]) {
    encode_u32(buf, ops.len() as u32);
    for op in ops {
        encode_operand(buf, op);
    }
}

fn decode_operands(data: &[u8], pos: &mut usize) -> Result<Vec<Operand>, TmirBinaryError> {
    let count = decode_u32(data, pos)? as usize;
    let mut result = Vec::with_capacity(count);
    for _ in 0..count {
        result.push(decode_operand(data, pos)?);
    }
    Ok(result)
}

fn encode_constant(buf: &mut Vec<u8>, c: &Constant) {
    match c {
        Constant::Int { value, ty } => {
            encode_u8(buf, 0);
            encode_i128(buf, *value);
            encode_ty(buf, ty);
        }
        Constant::Float { value, ty } => {
            encode_u8(buf, 1);
            encode_f64(buf, *value);
            encode_ty(buf, ty);
        }
        Constant::Bool(b) => {
            encode_u8(buf, 2);
            encode_bool(buf, *b);
        }
        Constant::Unit => {
            encode_u8(buf, 3);
        }
    }
}

fn decode_constant(data: &[u8], pos: &mut usize) -> Result<Constant, TmirBinaryError> {
    let tag = decode_u8(data, pos)?;
    match tag {
        0 => {
            let value = decode_i128(data, pos)?;
            let ty = decode_ty(data, pos)?;
            Ok(Constant::Int { value, ty })
        }
        1 => {
            let value = decode_f64(data, pos)?;
            let ty = decode_ty(data, pos)?;
            Ok(Constant::Float { value, ty })
        }
        2 => {
            let b = decode_bool(data, pos)?;
            Ok(Constant::Bool(b))
        }
        3 => Ok(Constant::Unit),
        _ => Err(TmirBinaryError::UnknownTag("Constant", tag)),
    }
}

// ---------------------------------------------------------------------------
// Instruction encoding/decoding
// ---------------------------------------------------------------------------

use tmir_instrs::{
    AtomicRmwOp, BinOp, CastOp, CmpOp, Instr, InstrNode, MemoryOrdering, SwitchCase, UnOp,
};

// Opcode tags for Instr variants — deterministic mapping.
const OP_BINOP: u8 = 0;
const OP_UNOP: u8 = 1;
const OP_CMP: u8 = 2;
const OP_CAST: u8 = 3;
const OP_LOAD: u8 = 4;
const OP_STORE: u8 = 5;
const OP_ALLOC: u8 = 6;
const OP_DEALLOC: u8 = 7;
const OP_ATOMIC_LOAD: u8 = 8;
const OP_ATOMIC_STORE: u8 = 9;
const OP_ATOMIC_RMW: u8 = 10;
const OP_CMPXCHG: u8 = 11;
const OP_FENCE: u8 = 12;
const OP_BORROW: u8 = 13;
const OP_BORROW_MUT: u8 = 14;
const OP_END_BORROW: u8 = 15;
const OP_RETAIN: u8 = 16;
const OP_RELEASE: u8 = 17;
const OP_IS_UNIQUE: u8 = 18;
const OP_BR: u8 = 19;
const OP_CONDBR: u8 = 20;
const OP_SWITCH: u8 = 21;
const OP_RETURN: u8 = 22;
const OP_CALL: u8 = 23;
const OP_CALL_INDIRECT: u8 = 24;
const OP_STRUCT: u8 = 25;
const OP_FIELD: u8 = 26;
const OP_INDEX: u8 = 27;
const OP_PHI: u8 = 28;
const OP_SELECT: u8 = 29;
const OP_GEP: u8 = 30;
const OP_NOP: u8 = 31;
const OP_CONST: u8 = 32;
const OP_FCONST: u8 = 33;

fn encode_binop_tag(buf: &mut Vec<u8>, op: BinOp) {
    encode_u8(buf, match op {
        BinOp::Add => 0, BinOp::Sub => 1, BinOp::Mul => 2, BinOp::SDiv => 3,
        BinOp::UDiv => 4, BinOp::SRem => 5, BinOp::URem => 6, BinOp::And => 7,
        BinOp::Or => 8, BinOp::Xor => 9, BinOp::Shl => 10, BinOp::AShr => 11,
        BinOp::LShr => 12, BinOp::FAdd => 13, BinOp::FSub => 14, BinOp::FMul => 15,
        BinOp::FDiv => 16,
    });
}

fn decode_binop_tag(data: &[u8], pos: &mut usize) -> Result<BinOp, TmirBinaryError> {
    let tag = decode_u8(data, pos)?;
    match tag {
        0 => Ok(BinOp::Add), 1 => Ok(BinOp::Sub), 2 => Ok(BinOp::Mul), 3 => Ok(BinOp::SDiv),
        4 => Ok(BinOp::UDiv), 5 => Ok(BinOp::SRem), 6 => Ok(BinOp::URem), 7 => Ok(BinOp::And),
        8 => Ok(BinOp::Or), 9 => Ok(BinOp::Xor), 10 => Ok(BinOp::Shl), 11 => Ok(BinOp::AShr),
        12 => Ok(BinOp::LShr), 13 => Ok(BinOp::FAdd), 14 => Ok(BinOp::FSub), 15 => Ok(BinOp::FMul),
        16 => Ok(BinOp::FDiv),
        _ => Err(TmirBinaryError::UnknownTag("BinOp", tag)),
    }
}

fn encode_unop_tag(buf: &mut Vec<u8>, op: UnOp) {
    encode_u8(buf, match op {
        UnOp::Neg => 0, UnOp::Not => 1, UnOp::FNeg => 2, UnOp::FAbs => 3, UnOp::FSqrt => 4,
    });
}

fn decode_unop_tag(data: &[u8], pos: &mut usize) -> Result<UnOp, TmirBinaryError> {
    let tag = decode_u8(data, pos)?;
    match tag {
        0 => Ok(UnOp::Neg), 1 => Ok(UnOp::Not), 2 => Ok(UnOp::FNeg),
        3 => Ok(UnOp::FAbs), 4 => Ok(UnOp::FSqrt),
        _ => Err(TmirBinaryError::UnknownTag("UnOp", tag)),
    }
}

fn encode_cmpop_tag(buf: &mut Vec<u8>, op: CmpOp) {
    encode_u8(buf, match op {
        CmpOp::Eq => 0, CmpOp::Ne => 1, CmpOp::Slt => 2, CmpOp::Sle => 3,
        CmpOp::Sgt => 4, CmpOp::Sge => 5, CmpOp::Ult => 6, CmpOp::Ule => 7,
        CmpOp::Ugt => 8, CmpOp::Uge => 9,
        CmpOp::FOeq => 10, CmpOp::FOne => 11, CmpOp::FOlt => 12, CmpOp::FOle => 13,
        CmpOp::FOgt => 14, CmpOp::FOge => 15,
        CmpOp::FUeq => 16, CmpOp::FUne => 17, CmpOp::FUlt => 18, CmpOp::FUle => 19,
        CmpOp::FUgt => 20, CmpOp::FUge => 21,
    });
}

fn decode_cmpop_tag(data: &[u8], pos: &mut usize) -> Result<CmpOp, TmirBinaryError> {
    let tag = decode_u8(data, pos)?;
    match tag {
        0 => Ok(CmpOp::Eq), 1 => Ok(CmpOp::Ne), 2 => Ok(CmpOp::Slt), 3 => Ok(CmpOp::Sle),
        4 => Ok(CmpOp::Sgt), 5 => Ok(CmpOp::Sge), 6 => Ok(CmpOp::Ult), 7 => Ok(CmpOp::Ule),
        8 => Ok(CmpOp::Ugt), 9 => Ok(CmpOp::Uge),
        10 => Ok(CmpOp::FOeq), 11 => Ok(CmpOp::FOne), 12 => Ok(CmpOp::FOlt), 13 => Ok(CmpOp::FOle),
        14 => Ok(CmpOp::FOgt), 15 => Ok(CmpOp::FOge),
        16 => Ok(CmpOp::FUeq), 17 => Ok(CmpOp::FUne), 18 => Ok(CmpOp::FUlt), 19 => Ok(CmpOp::FUle),
        20 => Ok(CmpOp::FUgt), 21 => Ok(CmpOp::FUge),
        _ => Err(TmirBinaryError::UnknownTag("CmpOp", tag)),
    }
}

fn encode_castop_tag(buf: &mut Vec<u8>, op: CastOp) {
    encode_u8(buf, match op {
        CastOp::ZExt => 0, CastOp::SExt => 1, CastOp::Trunc => 2,
        CastOp::FPToSI => 3, CastOp::FPToUI => 4, CastOp::SIToFP => 5,
        CastOp::UIToFP => 6, CastOp::FPExt => 7, CastOp::FPTrunc => 8,
        CastOp::PtrToInt => 9, CastOp::IntToPtr => 10, CastOp::Bitcast => 11,
    });
}

fn decode_castop_tag(data: &[u8], pos: &mut usize) -> Result<CastOp, TmirBinaryError> {
    let tag = decode_u8(data, pos)?;
    match tag {
        0 => Ok(CastOp::ZExt), 1 => Ok(CastOp::SExt), 2 => Ok(CastOp::Trunc),
        3 => Ok(CastOp::FPToSI), 4 => Ok(CastOp::FPToUI), 5 => Ok(CastOp::SIToFP),
        6 => Ok(CastOp::UIToFP), 7 => Ok(CastOp::FPExt), 8 => Ok(CastOp::FPTrunc),
        9 => Ok(CastOp::PtrToInt), 10 => Ok(CastOp::IntToPtr), 11 => Ok(CastOp::Bitcast),
        _ => Err(TmirBinaryError::UnknownTag("CastOp", tag)),
    }
}

fn encode_memory_ordering(buf: &mut Vec<u8>, ord: MemoryOrdering) {
    encode_u8(buf, match ord {
        MemoryOrdering::Relaxed => 0,
        MemoryOrdering::Acquire => 1,
        MemoryOrdering::Release => 2,
        MemoryOrdering::AcqRel => 3,
        MemoryOrdering::SeqCst => 4,
    });
}

fn decode_memory_ordering(data: &[u8], pos: &mut usize) -> Result<MemoryOrdering, TmirBinaryError> {
    let tag = decode_u8(data, pos)?;
    match tag {
        0 => Ok(MemoryOrdering::Relaxed),
        1 => Ok(MemoryOrdering::Acquire),
        2 => Ok(MemoryOrdering::Release),
        3 => Ok(MemoryOrdering::AcqRel),
        4 => Ok(MemoryOrdering::SeqCst),
        _ => Err(TmirBinaryError::UnknownTag("MemoryOrdering", tag)),
    }
}

fn encode_atomic_rmw_op(buf: &mut Vec<u8>, op: AtomicRmwOp) {
    encode_u8(buf, match op {
        AtomicRmwOp::Add => 0, AtomicRmwOp::Sub => 1, AtomicRmwOp::And => 2,
        AtomicRmwOp::Or => 3, AtomicRmwOp::Xor => 4, AtomicRmwOp::Xchg => 5,
    });
}

fn decode_atomic_rmw_op(data: &[u8], pos: &mut usize) -> Result<AtomicRmwOp, TmirBinaryError> {
    let tag = decode_u8(data, pos)?;
    match tag {
        0 => Ok(AtomicRmwOp::Add), 1 => Ok(AtomicRmwOp::Sub), 2 => Ok(AtomicRmwOp::And),
        3 => Ok(AtomicRmwOp::Or), 4 => Ok(AtomicRmwOp::Xor), 5 => Ok(AtomicRmwOp::Xchg),
        _ => Err(TmirBinaryError::UnknownTag("AtomicRmwOp", tag)),
    }
}

fn encode_value_ids(buf: &mut Vec<u8>, ids: &[ValueId]) {
    encode_u32(buf, ids.len() as u32);
    for id in ids {
        encode_u32(buf, id.0);
    }
}

fn decode_value_ids(data: &[u8], pos: &mut usize) -> Result<Vec<ValueId>, TmirBinaryError> {
    let count = decode_u32(data, pos)? as usize;
    let mut result = Vec::with_capacity(count);
    for _ in 0..count {
        result.push(ValueId(decode_u32(data, pos)?));
    }
    Ok(result)
}

fn encode_tys(buf: &mut Vec<u8>, tys: &[Ty]) {
    encode_u32(buf, tys.len() as u32);
    for ty in tys {
        encode_ty(buf, ty);
    }
}

fn decode_tys(data: &[u8], pos: &mut usize) -> Result<Vec<Ty>, TmirBinaryError> {
    let count = decode_u32(data, pos)? as usize;
    let mut result = Vec::with_capacity(count);
    for _ in 0..count {
        result.push(decode_ty(data, pos)?);
    }
    Ok(result)
}

fn encode_instr(buf: &mut Vec<u8>, instr: &Instr) {
    match instr {
        Instr::BinOp { op, ty, lhs, rhs } => {
            encode_u8(buf, OP_BINOP);
            encode_binop_tag(buf, *op);
            encode_ty(buf, ty);
            encode_operand(buf, lhs);
            encode_operand(buf, rhs);
        }
        Instr::UnOp { op, ty, operand } => {
            encode_u8(buf, OP_UNOP);
            encode_unop_tag(buf, *op);
            encode_ty(buf, ty);
            encode_operand(buf, operand);
        }
        Instr::Cmp { op, ty, lhs, rhs } => {
            encode_u8(buf, OP_CMP);
            encode_cmpop_tag(buf, *op);
            encode_ty(buf, ty);
            encode_operand(buf, lhs);
            encode_operand(buf, rhs);
        }
        Instr::Cast { op, src_ty, dst_ty, operand } => {
            encode_u8(buf, OP_CAST);
            encode_castop_tag(buf, *op);
            encode_ty(buf, src_ty);
            encode_ty(buf, dst_ty);
            encode_operand(buf, operand);
        }
        Instr::Load { ty, ptr } => {
            encode_u8(buf, OP_LOAD);
            encode_ty(buf, ty);
            encode_u32(buf, ptr.0);
        }
        Instr::Store { ty, ptr, value } => {
            encode_u8(buf, OP_STORE);
            encode_ty(buf, ty);
            encode_u32(buf, ptr.0);
            encode_operand(buf, value);
        }
        Instr::Alloc { ty, count } => {
            encode_u8(buf, OP_ALLOC);
            encode_ty(buf, ty);
            encode_bool(buf, count.is_some());
            if let Some(c) = count {
                encode_u32(buf, c.0);
            }
        }
        Instr::Dealloc { ptr } => {
            encode_u8(buf, OP_DEALLOC);
            encode_u32(buf, ptr.0);
        }
        Instr::AtomicLoad { ty, ptr, ordering } => {
            encode_u8(buf, OP_ATOMIC_LOAD);
            encode_ty(buf, ty);
            encode_u32(buf, ptr.0);
            encode_memory_ordering(buf, *ordering);
        }
        Instr::AtomicStore { ty, ptr, value, ordering } => {
            encode_u8(buf, OP_ATOMIC_STORE);
            encode_ty(buf, ty);
            encode_u32(buf, ptr.0);
            encode_operand(buf, value);
            encode_memory_ordering(buf, *ordering);
        }
        Instr::AtomicRmw { op, ty, ptr, value, ordering } => {
            encode_u8(buf, OP_ATOMIC_RMW);
            encode_atomic_rmw_op(buf, *op);
            encode_ty(buf, ty);
            encode_u32(buf, ptr.0);
            encode_operand(buf, value);
            encode_memory_ordering(buf, *ordering);
        }
        Instr::CmpXchg { ty, ptr, expected, desired, success_ordering, failure_ordering } => {
            encode_u8(buf, OP_CMPXCHG);
            encode_ty(buf, ty);
            encode_u32(buf, ptr.0);
            encode_operand(buf, expected);
            encode_operand(buf, desired);
            encode_memory_ordering(buf, *success_ordering);
            encode_memory_ordering(buf, *failure_ordering);
        }
        Instr::Fence { ordering } => {
            encode_u8(buf, OP_FENCE);
            encode_memory_ordering(buf, *ordering);
        }
        Instr::Borrow { ty, value } => {
            encode_u8(buf, OP_BORROW);
            encode_ty(buf, ty);
            encode_u32(buf, value.0);
        }
        Instr::BorrowMut { ty, value } => {
            encode_u8(buf, OP_BORROW_MUT);
            encode_ty(buf, ty);
            encode_u32(buf, value.0);
        }
        Instr::EndBorrow { borrow } => {
            encode_u8(buf, OP_END_BORROW);
            encode_u32(buf, borrow.0);
        }
        Instr::Retain { value } => {
            encode_u8(buf, OP_RETAIN);
            encode_u32(buf, value.0);
        }
        Instr::Release { value } => {
            encode_u8(buf, OP_RELEASE);
            encode_u32(buf, value.0);
        }
        Instr::IsUnique { value } => {
            encode_u8(buf, OP_IS_UNIQUE);
            encode_u32(buf, value.0);
        }
        Instr::Br { target, args } => {
            encode_u8(buf, OP_BR);
            encode_u32(buf, target.0);
            encode_operands(buf, args);
        }
        Instr::CondBr { cond, then_target, then_args, else_target, else_args } => {
            encode_u8(buf, OP_CONDBR);
            encode_operand(buf, cond);
            encode_u32(buf, then_target.0);
            encode_operands(buf, then_args);
            encode_u32(buf, else_target.0);
            encode_operands(buf, else_args);
        }
        Instr::Switch { value, cases, default } => {
            encode_u8(buf, OP_SWITCH);
            encode_operand(buf, value);
            encode_u32(buf, cases.len() as u32);
            for c in cases {
                encode_i64(buf, c.value);
                encode_u32(buf, c.target.0);
            }
            encode_u32(buf, default.0);
        }
        Instr::Return { values } => {
            encode_u8(buf, OP_RETURN);
            encode_operands(buf, values);
        }
        Instr::Call { func, args, ret_ty } => {
            encode_u8(buf, OP_CALL);
            encode_u32(buf, func.0);
            encode_operands(buf, args);
            encode_tys(buf, ret_ty);
        }
        Instr::CallIndirect { callee, args, ret_ty } => {
            encode_u8(buf, OP_CALL_INDIRECT);
            encode_u32(buf, callee.0);
            encode_operands(buf, args);
            encode_tys(buf, ret_ty);
        }
        Instr::Struct { ty, fields } => {
            encode_u8(buf, OP_STRUCT);
            encode_ty(buf, ty);
            encode_operands(buf, fields);
        }
        Instr::Field { ty, value, index } => {
            encode_u8(buf, OP_FIELD);
            encode_ty(buf, ty);
            encode_u32(buf, value.0);
            encode_u32(buf, *index);
        }
        Instr::Index { ty, base, index } => {
            encode_u8(buf, OP_INDEX);
            encode_ty(buf, ty);
            encode_u32(buf, base.0);
            encode_operand(buf, index);
        }
        Instr::Phi { ty, incoming } => {
            encode_u8(buf, OP_PHI);
            encode_ty(buf, ty);
            encode_u32(buf, incoming.len() as u32);
            for (block, op) in incoming {
                encode_u32(buf, block.0);
                encode_operand(buf, op);
            }
        }
        Instr::Select { ty, cond, true_val, false_val } => {
            encode_u8(buf, OP_SELECT);
            encode_ty(buf, ty);
            encode_operand(buf, cond);
            encode_operand(buf, true_val);
            encode_operand(buf, false_val);
        }
        Instr::GetElementPtr { elem_ty, base, index, offset } => {
            encode_u8(buf, OP_GEP);
            encode_ty(buf, elem_ty);
            encode_u32(buf, base.0);
            encode_operand(buf, index);
            encode_i32(buf, *offset);
        }
        Instr::Nop => {
            encode_u8(buf, OP_NOP);
        }
        Instr::Const { ty, value } => {
            encode_u8(buf, OP_CONST);
            encode_ty(buf, ty);
            encode_i64(buf, *value);
        }
        Instr::FConst { ty, value } => {
            encode_u8(buf, OP_FCONST);
            encode_ty(buf, ty);
            encode_f64(buf, *value);
        }
    }
}

fn decode_instr(data: &[u8], pos: &mut usize) -> Result<Instr, TmirBinaryError> {
    let opcode = decode_u8(data, pos)?;
    match opcode {
        OP_BINOP => {
            let op = decode_binop_tag(data, pos)?;
            let ty = decode_ty(data, pos)?;
            let lhs = decode_operand(data, pos)?;
            let rhs = decode_operand(data, pos)?;
            Ok(Instr::BinOp { op, ty, lhs, rhs })
        }
        OP_UNOP => {
            let op = decode_unop_tag(data, pos)?;
            let ty = decode_ty(data, pos)?;
            let operand = decode_operand(data, pos)?;
            Ok(Instr::UnOp { op, ty, operand })
        }
        OP_CMP => {
            let op = decode_cmpop_tag(data, pos)?;
            let ty = decode_ty(data, pos)?;
            let lhs = decode_operand(data, pos)?;
            let rhs = decode_operand(data, pos)?;
            Ok(Instr::Cmp { op, ty, lhs, rhs })
        }
        OP_CAST => {
            let op = decode_castop_tag(data, pos)?;
            let src_ty = decode_ty(data, pos)?;
            let dst_ty = decode_ty(data, pos)?;
            let operand = decode_operand(data, pos)?;
            Ok(Instr::Cast { op, src_ty, dst_ty, operand })
        }
        OP_LOAD => {
            let ty = decode_ty(data, pos)?;
            let ptr = ValueId(decode_u32(data, pos)?);
            Ok(Instr::Load { ty, ptr })
        }
        OP_STORE => {
            let ty = decode_ty(data, pos)?;
            let ptr = ValueId(decode_u32(data, pos)?);
            let value = decode_operand(data, pos)?;
            Ok(Instr::Store { ty, ptr, value })
        }
        OP_ALLOC => {
            let ty = decode_ty(data, pos)?;
            let has_count = decode_bool(data, pos)?;
            let count = if has_count {
                Some(ValueId(decode_u32(data, pos)?))
            } else {
                None
            };
            Ok(Instr::Alloc { ty, count })
        }
        OP_DEALLOC => {
            let ptr = ValueId(decode_u32(data, pos)?);
            Ok(Instr::Dealloc { ptr })
        }
        OP_ATOMIC_LOAD => {
            let ty = decode_ty(data, pos)?;
            let ptr = ValueId(decode_u32(data, pos)?);
            let ordering = decode_memory_ordering(data, pos)?;
            Ok(Instr::AtomicLoad { ty, ptr, ordering })
        }
        OP_ATOMIC_STORE => {
            let ty = decode_ty(data, pos)?;
            let ptr = ValueId(decode_u32(data, pos)?);
            let value = decode_operand(data, pos)?;
            let ordering = decode_memory_ordering(data, pos)?;
            Ok(Instr::AtomicStore { ty, ptr, value, ordering })
        }
        OP_ATOMIC_RMW => {
            let op = decode_atomic_rmw_op(data, pos)?;
            let ty = decode_ty(data, pos)?;
            let ptr = ValueId(decode_u32(data, pos)?);
            let value = decode_operand(data, pos)?;
            let ordering = decode_memory_ordering(data, pos)?;
            Ok(Instr::AtomicRmw { op, ty, ptr, value, ordering })
        }
        OP_CMPXCHG => {
            let ty = decode_ty(data, pos)?;
            let ptr = ValueId(decode_u32(data, pos)?);
            let expected = decode_operand(data, pos)?;
            let desired = decode_operand(data, pos)?;
            let success_ordering = decode_memory_ordering(data, pos)?;
            let failure_ordering = decode_memory_ordering(data, pos)?;
            Ok(Instr::CmpXchg { ty, ptr, expected, desired, success_ordering, failure_ordering })
        }
        OP_FENCE => {
            let ordering = decode_memory_ordering(data, pos)?;
            Ok(Instr::Fence { ordering })
        }
        OP_BORROW => {
            let ty = decode_ty(data, pos)?;
            let value = ValueId(decode_u32(data, pos)?);
            Ok(Instr::Borrow { ty, value })
        }
        OP_BORROW_MUT => {
            let ty = decode_ty(data, pos)?;
            let value = ValueId(decode_u32(data, pos)?);
            Ok(Instr::BorrowMut { ty, value })
        }
        OP_END_BORROW => {
            let borrow = ValueId(decode_u32(data, pos)?);
            Ok(Instr::EndBorrow { borrow })
        }
        OP_RETAIN => {
            let value = ValueId(decode_u32(data, pos)?);
            Ok(Instr::Retain { value })
        }
        OP_RELEASE => {
            let value = ValueId(decode_u32(data, pos)?);
            Ok(Instr::Release { value })
        }
        OP_IS_UNIQUE => {
            let value = ValueId(decode_u32(data, pos)?);
            Ok(Instr::IsUnique { value })
        }
        OP_BR => {
            let target = BlockId(decode_u32(data, pos)?);
            let args = decode_operands(data, pos)?;
            Ok(Instr::Br { target, args })
        }
        OP_CONDBR => {
            let cond = decode_operand(data, pos)?;
            let then_target = BlockId(decode_u32(data, pos)?);
            let then_args = decode_operands(data, pos)?;
            let else_target = BlockId(decode_u32(data, pos)?);
            let else_args = decode_operands(data, pos)?;
            Ok(Instr::CondBr { cond, then_target, then_args, else_target, else_args })
        }
        OP_SWITCH => {
            let value = decode_operand(data, pos)?;
            let count = decode_u32(data, pos)? as usize;
            let mut cases = Vec::with_capacity(count);
            for _ in 0..count {
                let val = decode_i64(data, pos)?;
                let target = BlockId(decode_u32(data, pos)?);
                cases.push(SwitchCase { value: val, target });
            }
            let default = BlockId(decode_u32(data, pos)?);
            Ok(Instr::Switch { value, cases, default })
        }
        OP_RETURN => {
            let values = decode_operands(data, pos)?;
            Ok(Instr::Return { values })
        }
        OP_CALL => {
            let func = FuncId(decode_u32(data, pos)?);
            let args = decode_operands(data, pos)?;
            let ret_ty = decode_tys(data, pos)?;
            Ok(Instr::Call { func, args, ret_ty })
        }
        OP_CALL_INDIRECT => {
            let callee = ValueId(decode_u32(data, pos)?);
            let args = decode_operands(data, pos)?;
            let ret_ty = decode_tys(data, pos)?;
            Ok(Instr::CallIndirect { callee, args, ret_ty })
        }
        OP_STRUCT => {
            let ty = decode_ty(data, pos)?;
            let fields = decode_operands(data, pos)?;
            Ok(Instr::Struct { ty, fields })
        }
        OP_FIELD => {
            let ty = decode_ty(data, pos)?;
            let value = ValueId(decode_u32(data, pos)?);
            let index = decode_u32(data, pos)?;
            Ok(Instr::Field { ty, value, index })
        }
        OP_INDEX => {
            let ty = decode_ty(data, pos)?;
            let base = ValueId(decode_u32(data, pos)?);
            let index = decode_operand(data, pos)?;
            Ok(Instr::Index { ty, base, index })
        }
        OP_PHI => {
            let ty = decode_ty(data, pos)?;
            let count = decode_u32(data, pos)? as usize;
            let mut incoming = Vec::with_capacity(count);
            for _ in 0..count {
                let block = BlockId(decode_u32(data, pos)?);
                let op = decode_operand(data, pos)?;
                incoming.push((block, op));
            }
            Ok(Instr::Phi { ty, incoming })
        }
        OP_SELECT => {
            let ty = decode_ty(data, pos)?;
            let cond = decode_operand(data, pos)?;
            let true_val = decode_operand(data, pos)?;
            let false_val = decode_operand(data, pos)?;
            Ok(Instr::Select { ty, cond, true_val, false_val })
        }
        OP_GEP => {
            let elem_ty = decode_ty(data, pos)?;
            let base = ValueId(decode_u32(data, pos)?);
            let index = decode_operand(data, pos)?;
            let offset = decode_i32(data, pos)?;
            Ok(Instr::GetElementPtr { elem_ty, base, index, offset })
        }
        OP_NOP => Ok(Instr::Nop),
        OP_CONST => {
            let ty = decode_ty(data, pos)?;
            let value = decode_i64(data, pos)?;
            Ok(Instr::Const { ty, value })
        }
        OP_FCONST => {
            let ty = decode_ty(data, pos)?;
            let value = decode_f64(data, pos)?;
            Ok(Instr::FConst { ty, value })
        }
        _ => Err(TmirBinaryError::UnknownOpcode(opcode)),
    }
}

fn encode_instr_node(buf: &mut Vec<u8>, node: &InstrNode) {
    encode_instr(buf, &node.instr);
    encode_value_ids(buf, &node.results);
    encode_proofs(buf, &node.proofs);
}

fn decode_instr_node(data: &[u8], pos: &mut usize) -> Result<InstrNode, TmirBinaryError> {
    let instr = decode_instr(data, pos)?;
    let results = decode_value_ids(data, pos)?;
    let proofs = decode_proofs(data, pos)?;
    Ok(InstrNode { instr, results, proofs })
}

// ---------------------------------------------------------------------------
// Block encoding/decoding
// ---------------------------------------------------------------------------

use crate::Block;

fn encode_block(buf: &mut Vec<u8>, block: &Block) {
    encode_u32(buf, block.id.0);
    // Params: (ValueId, Ty) pairs
    encode_u32(buf, block.params.len() as u32);
    for (vid, ty) in &block.params {
        encode_u32(buf, vid.0);
        encode_ty(buf, ty);
    }
    // Body
    encode_u32(buf, block.body.len() as u32);
    for node in &block.body {
        encode_instr_node(buf, node);
    }
}

fn decode_block(data: &[u8], pos: &mut usize) -> Result<Block, TmirBinaryError> {
    let id = BlockId(decode_u32(data, pos)?);
    let pcount = decode_u32(data, pos)? as usize;
    let mut params = Vec::with_capacity(pcount);
    for _ in 0..pcount {
        let vid = ValueId(decode_u32(data, pos)?);
        let ty = decode_ty(data, pos)?;
        params.push((vid, ty));
    }
    let bcount = decode_u32(data, pos)? as usize;
    let mut body = Vec::with_capacity(bcount);
    for _ in 0..bcount {
        body.push(decode_instr_node(data, pos)?);
    }
    Ok(Block { id, params, body })
}

// ---------------------------------------------------------------------------
// Section encoding helpers
// ---------------------------------------------------------------------------

/// Encode a section: section_id + payload_size + payload bytes.
fn encode_section(buf: &mut Vec<u8>, section_id: u8, payload: &[u8]) {
    encode_u8(buf, section_id);
    encode_u32(buf, payload.len() as u32);
    buf.extend_from_slice(payload);
}

// ---------------------------------------------------------------------------
// StructDef encoding/decoding
// ---------------------------------------------------------------------------

fn encode_struct_def(buf: &mut Vec<u8>, sd: &StructDef) {
    encode_u32(buf, sd.id.0);
    encode_string(buf, &sd.name);
    encode_u32(buf, sd.fields.len() as u32);
    for f in &sd.fields {
        encode_string(buf, &f.name);
        encode_ty(buf, &f.ty);
        encode_bool(buf, f.offset.is_some());
        if let Some(off) = f.offset {
            encode_u32(buf, off);
        }
    }
    encode_bool(buf, sd.size.is_some());
    if let Some(s) = sd.size {
        encode_u32(buf, s);
    }
    encode_bool(buf, sd.align.is_some());
    if let Some(a) = sd.align {
        encode_u32(buf, a);
    }
}

fn decode_struct_def(data: &[u8], pos: &mut usize) -> Result<StructDef, TmirBinaryError> {
    let id = StructId(decode_u32(data, pos)?);
    let name = decode_string(data, pos)?;
    let fcount = decode_u32(data, pos)? as usize;
    let mut fields = Vec::with_capacity(fcount);
    for _ in 0..fcount {
        let fname = decode_string(data, pos)?;
        let fty = decode_ty(data, pos)?;
        let has_offset = decode_bool(data, pos)?;
        let offset = if has_offset { Some(decode_u32(data, pos)?) } else { None };
        fields.push(FieldDef { name: fname, ty: fty, offset });
    }
    let has_size = decode_bool(data, pos)?;
    let size = if has_size { Some(decode_u32(data, pos)?) } else { None };
    let has_align = decode_bool(data, pos)?;
    let align = if has_align { Some(decode_u32(data, pos)?) } else { None };
    Ok(StructDef { id, name, fields, size, align })
}

// ---------------------------------------------------------------------------
// GlobalDef encoding/decoding
// ---------------------------------------------------------------------------

fn encode_global_def(buf: &mut Vec<u8>, g: &GlobalDef) {
    encode_string(buf, &g.name);
    encode_ty(buf, &g.ty);
    encode_bool(buf, g.is_const);
    encode_linkage(buf, g.linkage);
    encode_bool(buf, g.initializer.is_some());
    if let Some(init) = &g.initializer {
        encode_bytes(buf, init);
    }
    encode_bool(buf, g.align.is_some());
    if let Some(a) = g.align {
        encode_u32(buf, a);
    }
}

fn decode_global_def(data: &[u8], pos: &mut usize) -> Result<GlobalDef, TmirBinaryError> {
    let name = decode_string(data, pos)?;
    let ty = decode_ty(data, pos)?;
    let is_const = decode_bool(data, pos)?;
    let linkage = decode_linkage(data, pos)?;
    let has_init = decode_bool(data, pos)?;
    let initializer = if has_init { Some(decode_byte_vec(data, pos)?) } else { None };
    let has_align = decode_bool(data, pos)?;
    let align = if has_align { Some(decode_u32(data, pos)?) } else { None };
    Ok(GlobalDef { name, ty, is_const, linkage, initializer, align })
}

// ---------------------------------------------------------------------------
// DataLayout encoding/decoding
// ---------------------------------------------------------------------------

fn encode_data_layout(buf: &mut Vec<u8>, dl: &DataLayout) {
    encode_u32(buf, dl.pointer_size);
    encode_u32(buf, dl.pointer_align);
    encode_u32(buf, dl.stack_align);
    encode_bool(buf, dl.big_endian);
    encode_u32(buf, dl.int_align.len() as u32);
    for (bits, align) in &dl.int_align {
        encode_u32(buf, *bits as u32);
        encode_u32(buf, *align);
    }
}

fn decode_data_layout(data: &[u8], pos: &mut usize) -> Result<DataLayout, TmirBinaryError> {
    let pointer_size = decode_u32(data, pos)?;
    let pointer_align = decode_u32(data, pos)?;
    let stack_align = decode_u32(data, pos)?;
    let big_endian = decode_bool(data, pos)?;
    let count = decode_u32(data, pos)? as usize;
    let mut int_align = Vec::with_capacity(count);
    for _ in 0..count {
        let bits = decode_u32(data, pos)? as u16;
        let align = decode_u32(data, pos)?;
        int_align.push((bits, align));
    }
    Ok(DataLayout { pointer_size, pointer_align, stack_align, big_endian, int_align })
}

// ---------------------------------------------------------------------------
// Function encoding/decoding (header + body)
// ---------------------------------------------------------------------------

use crate::Function;

fn encode_function_header(buf: &mut Vec<u8>, func: &Function) {
    encode_u32(buf, func.id.0);
    encode_string(buf, &func.name);
    encode_func_ty(buf, &func.ty);
    encode_u32(buf, func.entry.0);
    encode_proofs(buf, &func.proofs);
}

fn decode_function_header(data: &[u8], pos: &mut usize) -> Result<(FuncId, String, FuncTy, BlockId, Vec<TmirProof>), TmirBinaryError> {
    let id = FuncId(decode_u32(data, pos)?);
    let name = decode_string(data, pos)?;
    let ty = decode_func_ty(data, pos)?;
    let entry = BlockId(decode_u32(data, pos)?);
    let proofs = decode_proofs(data, pos)?;
    Ok((id, name, ty, entry, proofs))
}

fn encode_function_body(buf: &mut Vec<u8>, func: &Function) {
    encode_u32(buf, func.blocks.len() as u32);
    for block in &func.blocks {
        encode_block(buf, block);
    }
}

fn decode_function_body(data: &[u8], pos: &mut usize) -> Result<Vec<Block>, TmirBinaryError> {
    let count = decode_u32(data, pos)? as usize;
    let mut blocks = Vec::with_capacity(count);
    for _ in 0..count {
        blocks.push(decode_block(data, pos)?);
    }
    Ok(blocks)
}

// ---------------------------------------------------------------------------
// Public API: encode/decode module
// ---------------------------------------------------------------------------

/// Encode a tMIR module to binary (.tmbc) format.
///
/// The output is a deterministic, canonical byte sequence:
///   magic + version + type_section + struct_section + global_section +
///   function_section + code_section + data_layout_section (optional)
pub fn write_module_to_binary(module: &Module) -> Vec<u8> {
    let mut buf = Vec::new();

    // Magic + version
    buf.extend_from_slice(MAGIC);
    encode_u32(&mut buf, VERSION);

    // Module name (in a pseudo-section at the start, as part of the header)
    encode_string(&mut buf, &module.name);

    // Section 2: Struct definitions
    if !module.structs.is_empty() {
        let mut payload = Vec::new();
        encode_u32(&mut payload, module.structs.len() as u32);
        for sd in &module.structs {
            encode_struct_def(&mut payload, sd);
        }
        encode_section(&mut buf, SECTION_STRUCT, &payload);
    }

    // Section 3: Global definitions
    if !module.globals.is_empty() {
        let mut payload = Vec::new();
        encode_u32(&mut payload, module.globals.len() as u32);
        for g in &module.globals {
            encode_global_def(&mut payload, g);
        }
        encode_section(&mut buf, SECTION_GLOBAL, &payload);
    }

    // Section 4: Function headers
    if !module.functions.is_empty() {
        let mut payload = Vec::new();
        encode_u32(&mut payload, module.functions.len() as u32);
        for func in &module.functions {
            encode_function_header(&mut payload, func);
        }
        encode_section(&mut buf, SECTION_FUNCTION, &payload);
    }

    // Section 5: Code (function bodies)
    if !module.functions.is_empty() {
        let mut payload = Vec::new();
        encode_u32(&mut payload, module.functions.len() as u32);
        for func in &module.functions {
            encode_function_body(&mut payload, func);
        }
        encode_section(&mut buf, SECTION_CODE, &payload);
    }

    // Section 6: Data layout (optional)
    if let Some(dl) = &module.data_layout {
        let mut payload = Vec::new();
        encode_data_layout(&mut payload, dl);
        encode_section(&mut buf, SECTION_DATA_LAYOUT, &payload);
    }

    buf
}

/// Decode a tMIR module from binary (.tmbc) format.
pub fn read_module_from_binary(bytes: &[u8]) -> Result<Module, TmirBinaryError> {
    // Magic
    if bytes.len() < 4 {
        return Err(TmirBinaryError::UnexpectedEof);
    }
    let magic: [u8; 4] = bytes[0..4].try_into().unwrap();
    let mut pos = 4;
    if &magic != MAGIC {
        return Err(TmirBinaryError::InvalidMagic(magic));
    }

    // Version
    let version = decode_u32(bytes, &mut pos)?;
    if version != VERSION {
        return Err(TmirBinaryError::UnsupportedVersion(version));
    }

    // Module name
    let name = decode_string(bytes, &mut pos)?;

    // Initialize module fields
    let mut structs = Vec::new();
    let mut globals = Vec::new();
    let mut function_headers: Vec<(FuncId, String, FuncTy, BlockId, Vec<TmirProof>)> = Vec::new();
    let mut function_bodies: Vec<Vec<Block>> = Vec::new();
    let mut data_layout = None;

    // Parse sections
    while pos < bytes.len() {
        let section_id = decode_u8(bytes, &mut pos)?;
        let payload_size = decode_u32(bytes, &mut pos)? as usize;
        if pos + payload_size > bytes.len() {
            return Err(TmirBinaryError::UnexpectedEof);
        }
        let section_end = pos + payload_size;

        match section_id {
            SECTION_TYPE => {
                // Reserved for future type table deduplication.
                // For now, skip the payload.
                pos = section_end;
            }
            SECTION_STRUCT => {
                let count = decode_u32(bytes, &mut pos)? as usize;
                for _ in 0..count {
                    structs.push(decode_struct_def(bytes, &mut pos)?);
                }
            }
            SECTION_GLOBAL => {
                let count = decode_u32(bytes, &mut pos)? as usize;
                for _ in 0..count {
                    globals.push(decode_global_def(bytes, &mut pos)?);
                }
            }
            SECTION_FUNCTION => {
                let count = decode_u32(bytes, &mut pos)? as usize;
                for _ in 0..count {
                    function_headers.push(decode_function_header(bytes, &mut pos)?);
                }
            }
            SECTION_CODE => {
                let count = decode_u32(bytes, &mut pos)? as usize;
                for _ in 0..count {
                    function_bodies.push(decode_function_body(bytes, &mut pos)?);
                }
            }
            SECTION_DATA_LAYOUT => {
                data_layout = Some(decode_data_layout(bytes, &mut pos)?);
            }
            0 => {
                // Custom section: skip
                pos = section_end;
            }
            _ => {
                // Unknown section: skip for forward compatibility
                pos = section_end;
            }
        }
    }

    // Assemble functions from headers + bodies
    if function_headers.len() != function_bodies.len() {
        return Err(TmirBinaryError::Validation(format!(
            "function header count ({}) does not match code body count ({})",
            function_headers.len(),
            function_bodies.len()
        )));
    }

    let mut functions = Vec::with_capacity(function_headers.len());
    for ((id, fname, ty, entry, proofs), blocks) in
        function_headers.into_iter().zip(function_bodies.into_iter())
    {
        functions.push(Function { id, name: fname, ty, entry, blocks, proofs });
    }

    Ok(Module { name, functions, structs, globals, data_layout })
}

/// Round-trip test helper: encode then decode.
pub fn round_trip_binary(module: &Module) -> Result<Module, TmirBinaryError> {
    let bytes = write_module_to_binary(module);
    read_module_from_binary(&bytes)
}

// ---------------------------------------------------------------------------
// File I/O convenience wrappers
// ---------------------------------------------------------------------------

/// Returns `true` if `bytes` starts with the tMBC magic header.
pub fn is_tmbc_format(bytes: &[u8]) -> bool {
    bytes.len() >= 4 && bytes[..4] == *MAGIC
}

/// Read a tMIR module from a `.tmbc` file on disk.
pub fn read_module_from_tmbc(path: &std::path::Path) -> Result<Module, TmirBinaryError> {
    let bytes = std::fs::read(path).map_err(TmirBinaryError::IoError)?;
    read_module_from_binary(&bytes)
}

/// Serialize a tMIR module to binary and write it to a `.tmbc` file on disk.
pub fn write_module_to_tmbc(
    module: &Module,
    path: &std::path::Path,
) -> Result<(), TmirBinaryError> {
    let bytes = write_module_to_binary(module);
    std::fs::write(path, bytes).map_err(TmirBinaryError::IoError)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::{self, ModuleBuilder};
    use tmir_instrs::{BinOp, CastOp, CmpOp, UnOp};
    use tmir_types::{
        BlockId, DataLayout, FieldDef, FuncId, FuncTy, GlobalDef, Linkage, StructDef, StructId,
        TmirProof, Ty, ValueId,
    };

    // -- LEB128 unit tests --

    #[test]
    fn test_uleb128_round_trip() {
        let values = [0u64, 1, 127, 128, 255, 256, 16384, u32::MAX as u64, u64::MAX];
        for &v in &values {
            let mut buf = Vec::new();
            encode_uleb128(&mut buf, v);
            let mut pos = 0;
            let decoded = decode_uleb128(&buf, &mut pos).unwrap();
            assert_eq!(v, decoded, "ULEB128 round-trip failed for {v}");
            assert_eq!(pos, buf.len());
        }
    }

    #[test]
    fn test_sleb128_round_trip() {
        let values: [i128; 9] = [0, 1, -1, 63, -64, 127, -128, i64::MAX as i128, i64::MIN as i128];
        for &v in &values {
            let mut buf = Vec::new();
            encode_sleb128(&mut buf, v);
            let mut pos = 0;
            let decoded = decode_sleb128(&buf, &mut pos).unwrap();
            assert_eq!(v, decoded, "SLEB128 round-trip failed for {v}");
            assert_eq!(pos, buf.len());
        }
    }

    #[test]
    fn test_sleb128_i128_extremes() {
        let values: [i128; 4] = [i128::MIN, i128::MAX, i128::MIN + 1, i128::MAX - 1];
        for &v in &values {
            let mut buf = Vec::new();
            encode_sleb128(&mut buf, v);
            let mut pos = 0;
            let decoded = decode_sleb128(&buf, &mut pos).unwrap();
            assert_eq!(v, decoded, "SLEB128 i128 round-trip failed for {v}");
        }
    }

    // -- Helper to build a simple add module --

    fn make_add_module() -> Module {
        let mut mb = ModuleBuilder::new("test_add");
        let mut fb = mb.function("add_i64", vec![Ty::i64(), Ty::i64()], vec![Ty::i64()]);
        let (entry_id, params) = fb.entry_block();
        let result = fb.fresh_value();
        fb.add_block(
            entry_id,
            vec![(params[0], Ty::i64()), (params[1], Ty::i64())],
            vec![
                builder::binop(BinOp::Add, Ty::i64(), params[0], params[1], result),
                builder::ret(vec![result]),
            ],
        );
        let func = fb.build();
        mb.add_function(func);
        mb.build()
    }

    // -- Test 1: Empty module round-trip --

    #[test]
    fn test_empty_module_round_trip() {
        let module = Module::new("empty");
        let decoded = round_trip_binary(&module).unwrap();
        assert_eq!(module, decoded);
    }

    // -- Test 2: Single function round-trip --

    #[test]
    fn test_single_function_round_trip() {
        let module = make_add_module();
        let decoded = round_trip_binary(&module).unwrap();
        assert_eq!(module, decoded);
    }

    // -- Test 3: Multi-function module --

    #[test]
    fn test_multi_function_round_trip() {
        let mut mb = ModuleBuilder::new("multi");

        // Function 0: add
        let mut fb0 = mb.function("add", vec![Ty::i32(), Ty::i32()], vec![Ty::i32()]);
        let (entry0, p0) = fb0.entry_block();
        let r0 = fb0.fresh_value();
        fb0.add_block(
            entry0,
            vec![(p0[0], Ty::i32()), (p0[1], Ty::i32())],
            vec![
                builder::binop(BinOp::Add, Ty::i32(), p0[0], p0[1], r0),
                builder::ret(vec![r0]),
            ],
        );
        mb.add_function(fb0.build());

        // Function 1: sub
        let mut fb1 = mb.function("sub", vec![Ty::i32(), Ty::i32()], vec![Ty::i32()]);
        let (entry1, p1) = fb1.entry_block();
        let r1 = fb1.fresh_value();
        fb1.add_block(
            entry1,
            vec![(p1[0], Ty::i32()), (p1[1], Ty::i32())],
            vec![
                builder::binop(BinOp::Sub, Ty::i32(), p1[0], p1[1], r1),
                builder::ret(vec![r1]),
            ],
        );
        mb.add_function(fb1.build());

        // Function 2: negate
        let mut fb2 = mb.function("neg", vec![Ty::i64()], vec![Ty::i64()]);
        let (entry2, p2) = fb2.entry_block();
        let r2 = fb2.fresh_value();
        fb2.add_block(
            entry2,
            vec![(p2[0], Ty::i64())],
            vec![
                builder::unop(UnOp::Neg, Ty::i64(), p2[0], r2),
                builder::ret(vec![r2]),
            ],
        );
        mb.add_function(fb2.build());

        let module = mb.build();
        let decoded = round_trip_binary(&module).unwrap();
        assert_eq!(module, decoded);
    }

    // -- Test 4: All scalar types --

    #[test]
    fn test_all_scalar_types_round_trip() {
        let types = vec![
            Ty::i8(), Ty::i16(), Ty::i32(), Ty::i64(), Ty::i128(),
            Ty::u8(), Ty::u16(), Ty::u32(), Ty::u64(),
            Ty::f32(), Ty::f64(),
            Ty::bool_ty(), Ty::unit(), Ty::never(),
        ];
        for ty in &types {
            let mut buf = Vec::new();
            encode_ty(&mut buf, ty);
            let mut pos = 0;
            let decoded = decode_ty(&buf, &mut pos).unwrap();
            assert_eq!(*ty, decoded, "Type round-trip failed for {:?}", ty);
        }
    }

    // -- Test 5: Complex types (array, tuple, struct, enum, ref, vector, fnptr) --

    #[test]
    fn test_complex_types_round_trip() {
        let types = vec![
            Ty::array(Ty::i32(), 10),
            Ty::Tuple(vec![Ty::i32(), Ty::f64(), Ty::bool_ty()]),
            Ty::StructDef {
                name: "Point".to_string(),
                fields: vec![
                    Field { name: "x".to_string(), ty: Ty::f64() },
                    Field { name: "y".to_string(), ty: Ty::f64() },
                ],
            },
            Ty::Enum {
                name: "Option".to_string(),
                variants: vec![
                    Variant { name: "None".to_string(), fields: vec![] },
                    Variant { name: "Some".to_string(), fields: vec![Ty::i64()] },
                ],
            },
            Ty::ptr(Ty::i32()),
            Ty::ptr_mut(Ty::i64()),
            Ty::borrow(Ty::f32()),
            Ty::borrow_mut(Ty::f64()),
            Ty::rc(Ty::i32()),
            Ty::vector(Ty::i32(), 4),
            Ty::vector(Ty::f64(), 2),
            Ty::FnPtr {
                params: vec![Ty::i32(), Ty::i32()],
                ret: Box::new(Ty::i64()),
            },
            Ty::Struct(StructId(42)),
            Ty::Func(FuncTy {
                params: vec![Ty::i32()],
                returns: vec![Ty::i64()],
            }),
        ];
        for ty in &types {
            let mut buf = Vec::new();
            encode_ty(&mut buf, ty);
            let mut pos = 0;
            let decoded = decode_ty(&buf, &mut pos).unwrap();
            assert_eq!(*ty, decoded, "Complex type round-trip failed for {:?}", ty);
        }
    }

    // -- Test 6: All instruction types --

    #[test]
    fn test_all_instruction_types_round_trip() {
        let v0 = ValueId(0);
        let v1 = ValueId(1);
        let v2 = ValueId(2);
        let b0 = BlockId(0);
        let b1 = BlockId(1);

        let instructions: Vec<InstrNode> = vec![
            builder::binop(BinOp::Add, Ty::i64(), v0, v1, v2),
            builder::binop(BinOp::FMul, Ty::f64(), v0, v1, v2),
            builder::unop(UnOp::Neg, Ty::i32(), v0, v1),
            builder::unop(UnOp::FSqrt, Ty::f64(), v0, v1),
            builder::cmp(CmpOp::Slt, Ty::i64(), v0, v1, v2),
            builder::cmp(CmpOp::FOeq, Ty::f64(), v0, v1, v2),
            builder::cast(CastOp::ZExt, Ty::i32(), Ty::i64(), v0, v1),
            builder::cast(CastOp::FPToSI, Ty::f64(), Ty::i64(), v0, v1),
            builder::load(Ty::i64(), v0, v1),
            builder::store(Ty::i64(), v0, v1),
            builder::alloc(Ty::i32(), v0),
            builder::iconst(Ty::i64(), 42, v0),
            builder::fconst(Ty::f64(), 3.14, v0),
            builder::br(b0, vec![v0]),
            builder::condbr(v0, b0, vec![v1], b1, vec![v2]),
            builder::ret(vec![v0]),
            builder::call(FuncId(0), vec![v0, v1], vec![Ty::i32()], vec![v2]),
            builder::call_indirect(v0, vec![v1], vec![Ty::i64()], vec![v2]),
            builder::switch(v0, vec![(1, b0), (2, b1)], BlockId(3)),
            builder::select(Ty::i32(), v0, v1, v2, ValueId(3)),
            builder::gep(Ty::i32(), v0, v1, 8, v2),
            builder::field(Ty::i32(), v0, 2, v1),
            builder::struct_val(Ty::i32(), vec![v0, v1], v2),
            builder::index(Ty::i32(), v0, v1, v2),
            builder::phi(Ty::i64(), vec![(b0, v0), (b1, v1)], v2),
            InstrNode::new(Instr::Nop, vec![]),
            builder::dealloc(v0),
            builder::borrow_val(Ty::ptr(Ty::i32()), v0, v1),
            builder::borrow_mut(Ty::ptr_mut(Ty::i64()), v0, v1),
            builder::end_borrow(v0),
            builder::retain(v0),
            builder::release(v0),
            builder::is_unique(v0, v1),
            builder::atomic_load(Ty::i64(), v0, MemoryOrdering::Acquire, v1),
            builder::atomic_store(Ty::i64(), v0, v1, MemoryOrdering::Release),
            builder::atomic_rmw(AtomicRmwOp::Add, Ty::i64(), v0, v1, MemoryOrdering::SeqCst, v2),
            builder::cmpxchg(Ty::i64(), v0, v1, v2, MemoryOrdering::AcqRel, MemoryOrdering::Acquire, vec![ValueId(3), ValueId(4)]),
            builder::fence(MemoryOrdering::SeqCst),
        ];

        for (i, node) in instructions.iter().enumerate() {
            let mut buf = Vec::new();
            encode_instr_node(&mut buf, node);
            let mut pos = 0;
            let decoded = decode_instr_node(&buf, &mut pos).unwrap();
            assert_eq!(*node, decoded, "Instruction round-trip failed at index {i}: {:?}", node.instr);
            assert_eq!(pos, buf.len(), "Not all bytes consumed for instruction at index {i}");
        }
    }

    // -- Test 7: Empty blocks --

    #[test]
    fn test_empty_blocks_round_trip() {
        let mut mb = ModuleBuilder::new("empty_blocks");
        let mut fb = mb.function("f", vec![], vec![]);
        let (entry, _) = fb.entry_block();
        // Entry block with no params, no body except a return
        fb.add_block(entry, vec![], vec![builder::ret(vec![])]);
        mb.add_function(fb.build());
        let module = mb.build();
        let decoded = round_trip_binary(&module).unwrap();
        assert_eq!(module, decoded);
    }

    // -- Test 8: Module with struct definitions and globals --

    #[test]
    fn test_structs_and_globals_round_trip() {
        let mut mb = ModuleBuilder::new("with_extras");
        mb.add_struct(StructDef {
            id: StructId(0),
            name: "Point".to_string(),
            fields: vec![
                FieldDef { name: "x".to_string(), ty: Ty::f64(), offset: Some(0) },
                FieldDef { name: "y".to_string(), ty: Ty::f64(), offset: Some(8) },
            ],
            size: Some(16),
            align: Some(8),
        });
        mb.add_global(GlobalDef {
            name: "_origin".to_string(),
            ty: Ty::Struct(StructId(0)),
            is_const: true,
            linkage: Linkage::External,
            initializer: Some(vec![0; 16]),
            align: Some(8),
        });
        mb.add_global(GlobalDef {
            name: "_counter".to_string(),
            ty: Ty::i64(),
            is_const: false,
            linkage: Linkage::Internal,
            initializer: None,
            align: None,
        });

        // Add a trivial function
        let mut fb = mb.function("get_origin_x", vec![], vec![Ty::f64()]);
        let (entry, _) = fb.entry_block();
        let r = fb.fresh_value();
        fb.add_block(entry, vec![], vec![
            builder::fconst(Ty::f64(), 0.0, r),
            builder::ret(vec![r]),
        ]);
        mb.add_function(fb.build());

        let module = mb.build();
        let decoded = round_trip_binary(&module).unwrap();
        assert_eq!(module, decoded);
    }

    // -- Test 9: Module with data layout --

    #[test]
    fn test_data_layout_round_trip() {
        let mut mb = ModuleBuilder::new("with_layout");
        mb.with_data_layout(DataLayout::aarch64());
        let mut fb = mb.function("nop", vec![], vec![]);
        let (entry, _) = fb.entry_block();
        fb.add_block(entry, vec![], vec![builder::ret(vec![])]);
        mb.add_function(fb.build());
        let module = mb.build();
        let decoded = round_trip_binary(&module).unwrap();
        assert_eq!(module, decoded);
    }

    // -- Test 10: Proof annotations round-trip --

    #[test]
    fn test_proof_annotations_round_trip() {
        let v0 = ValueId(0);
        let v1 = ValueId(1);

        let proofs = vec![
            TmirProof::Pure,
            TmirProof::ValidBorrow { borrow: v0 },
            TmirProof::InBounds { base: v0, index: v1 },
            TmirProof::Associative,
            TmirProof::Commutative,
            TmirProof::NoOverflow { signed: true },
            TmirProof::NoOverflow { signed: false },
            TmirProof::NotNull { ptr: v0 },
            TmirProof::NonZeroDivisor { divisor: v1 },
            TmirProof::ValidShift { amount: v0, bitwidth: 64 },
            TmirProof::InRange { lo: -128, hi: 127 },
            TmirProof::Idempotent,
        ];

        for (i, proof) in proofs.iter().enumerate() {
            let mut buf = Vec::new();
            encode_proof(&mut buf, proof);
            let mut pos = 0;
            let decoded = decode_proof(&buf, &mut pos).unwrap();
            assert_eq!(*proof, decoded, "Proof round-trip failed at index {i}");
        }
    }

    // -- Test 11: Function with proofs on instructions --

    #[test]
    fn test_function_with_instruction_proofs_round_trip() {
        let mut mb = ModuleBuilder::new("proven");
        let mut fb = mb.function("checked_add", vec![Ty::i64(), Ty::i64()], vec![Ty::i64()]);
        let fb_ref = &mut fb;
        let (entry, params) = fb_ref.entry_block();
        let result = fb_ref.fresh_value();
        let add_node = InstrNode::with_proofs(
            Instr::BinOp {
                op: BinOp::Add,
                ty: Ty::i64(),
                lhs: Operand::Value(params[0]),
                rhs: Operand::Value(params[1]),
            },
            vec![result],
            vec![TmirProof::NoOverflow { signed: true }, TmirProof::Commutative],
        );
        fb_ref.add_block(
            entry,
            vec![(params[0], Ty::i64()), (params[1], Ty::i64())],
            vec![add_node, builder::ret(vec![result])],
        );
        let func = fb.build().with_proof(TmirProof::Pure);
        mb.add_function(func);
        let module = mb.build();
        let decoded = round_trip_binary(&module).unwrap();
        assert_eq!(module, decoded);
    }

    // -- Test 12: Large constant values --

    #[test]
    fn test_large_constants_round_trip() {
        let mut mb = ModuleBuilder::new("big_consts");
        let mut fb = mb.function("big", vec![], vec![Ty::i128()]);
        let (entry, _) = fb.entry_block();
        let v0 = fb.fresh_value();
        let v1 = fb.fresh_value();
        let v2 = fb.fresh_value();

        // Use Operand::Constant with i128 extreme values
        let add_node = InstrNode::new(
            Instr::BinOp {
                op: BinOp::Add,
                ty: Ty::i128(),
                lhs: Operand::Constant(Constant::Int { value: i128::MAX, ty: Ty::i128() }),
                rhs: Operand::Constant(Constant::Int { value: i128::MIN, ty: Ty::i128() }),
            },
            vec![v2],
        );

        fb.add_block(entry, vec![], vec![
            builder::iconst(Ty::i64(), i64::MAX, v0),
            builder::iconst(Ty::i64(), i64::MIN, v1),
            add_node,
            builder::ret(vec![v2]),
        ]);
        mb.add_function(fb.build());
        let module = mb.build();
        let decoded = round_trip_binary(&module).unwrap();
        assert_eq!(module, decoded);
    }

    // -- Test 13: Deterministic encoding (same module = same bytes) --

    #[test]
    fn test_deterministic_encoding() {
        let module = make_add_module();
        let bytes1 = write_module_to_binary(&module);
        let bytes2 = write_module_to_binary(&module);
        assert_eq!(bytes1, bytes2, "Encoding is not deterministic");
    }

    // -- Test 14: Binary is smaller than JSON --

    #[test]
    fn test_binary_smaller_than_json() {
        let module = make_add_module();
        let binary = write_module_to_binary(&module);
        let json = serde_json::to_string(&module).unwrap();
        assert!(
            binary.len() < json.len(),
            "Binary ({} bytes) should be smaller than JSON ({} bytes)",
            binary.len(),
            json.len(),
        );
    }

    // -- Test 15: Invalid magic rejected --

    #[test]
    fn test_invalid_magic_rejected() {
        let bytes = b"BADMxxxxxxx";
        let result = read_module_from_binary(bytes);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("invalid magic"));
    }

    // -- Test 16: Truncated input rejected --

    #[test]
    fn test_truncated_input_rejected() {
        let module = make_add_module();
        let bytes = write_module_to_binary(&module);
        // Truncate to half
        let truncated = &bytes[..bytes.len() / 2];
        let result = read_module_from_binary(truncated);
        assert!(result.is_err());
    }

    // -- Test 17: Multi-block function with control flow --

    #[test]
    fn test_multi_block_control_flow_round_trip() {
        let mut mb = ModuleBuilder::new("control_flow");
        let mut fb = mb.function("abs", vec![Ty::i64()], vec![Ty::i64()]);
        let (entry, params) = fb.entry_block();
        let cmp_result = fb.fresh_value();
        let neg_result = fb.fresh_value();
        let then_block = fb.fresh_block();
        let else_block = fb.fresh_block();

        // Entry: compare n < 0
        fb.add_block(
            entry,
            vec![(params[0], Ty::i64())],
            vec![
                builder::cmp(CmpOp::Slt, Ty::i64(), params[0], ValueId(100), cmp_result),
                builder::condbr(cmp_result, then_block, vec![params[0]], else_block, vec![params[0]]),
            ],
        );

        // Then: negate
        let then_param = fb.fresh_value();
        fb.add_block(
            then_block,
            vec![(then_param, Ty::i64())],
            vec![
                builder::unop(UnOp::Neg, Ty::i64(), then_param, neg_result),
                builder::ret(vec![neg_result]),
            ],
        );

        // Else: return as-is
        let else_param = fb.fresh_value();
        fb.add_block(
            else_block,
            vec![(else_param, Ty::i64())],
            vec![builder::ret(vec![else_param])],
        );

        mb.add_function(fb.build());
        let module = mb.build();
        let decoded = round_trip_binary(&module).unwrap();
        assert_eq!(module, decoded);
    }

    // -- Test 18: Inline constant operands round-trip --

    #[test]
    fn test_inline_constant_operands_round_trip() {
        let mut mb = ModuleBuilder::new("inline_consts");
        let mut fb = mb.function("add_42", vec![Ty::i64()], vec![Ty::i64()]);
        let (entry, params) = fb.entry_block();
        let result = fb.fresh_value();

        // Use inline constant in rhs
        let add_node = InstrNode::new(
            Instr::BinOp {
                op: BinOp::Add,
                ty: Ty::i64(),
                lhs: Operand::Value(params[0]),
                rhs: Operand::Constant(Constant::Int { value: 42, ty: Ty::i64() }),
            },
            vec![result],
        );

        fb.add_block(
            entry,
            vec![(params[0], Ty::i64())],
            vec![add_node, builder::ret(vec![result])],
        );
        mb.add_function(fb.build());
        let module = mb.build();
        let decoded = round_trip_binary(&module).unwrap();
        assert_eq!(module, decoded);
    }

    // -- Test 19: f64 constant round-trip (NaN, infinity, negative zero) --

    #[test]
    fn test_f64_special_values_round_trip() {
        let values = [0.0f64, -0.0, f64::INFINITY, f64::NEG_INFINITY, 1.0, -1.0, std::f64::consts::PI];
        for &v in &values {
            let mut buf = Vec::new();
            encode_f64(&mut buf, v);
            let mut pos = 0;
            let decoded = decode_f64(&buf, &mut pos).unwrap();
            assert_eq!(v.to_bits(), decoded.to_bits(), "f64 round-trip failed for {v}");
        }
    }

    // -- Test 20: Version mismatch rejected --

    #[test]
    fn test_version_mismatch_rejected() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC);
        encode_u32(&mut bytes, 99); // bad version
        let result = read_module_from_binary(&bytes);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("unsupported version"));
    }
}
