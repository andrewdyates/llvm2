// llvm2-dialect - DialectOp + Attribute
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0

//! [`DialectOp`] and [`Attribute`]: the SSA op wrapper and its static metadata.

use llvm2_ir::Type;

use crate::id::{BlockId, DialectOpId, OpId, ValueId};

/// A minimal attribute value type. Attributes are static metadata attached to
/// a [`DialectOp`] (e.g. loop bounds, batch sizes, target hints). Richer
/// attribute kinds (dense arrays, symbol refs, nested dicts) are future work.
///
/// The [`Attribute::Block`] variant exists so control-flow ops (branches,
/// jumps, returns that fall through) can encode their successor block(s) as
/// attributes without inventing a separate `successors: Vec<BlockId>` field.
/// Callers can walk and remap block references centrally via
/// [`DialectOp::remap_block_refs`].
#[derive(Debug, Clone, PartialEq)]
pub enum Attribute {
    I64(i64),
    U64(u64),
    Bool(bool),
    Str(String),
    /// Reference to a basic block inside the owning [`DialectFunction`].
    /// Conversion infrastructure (`ConversionDriver`) automatically remaps
    /// these through the source→destination block map.
    Block(BlockId),
    List(Vec<Attribute>),
}

impl Attribute {
    pub fn as_i64(&self) -> Option<i64> {
        if let Attribute::I64(v) = self { Some(*v) } else { None }
    }
    pub fn as_u64(&self) -> Option<u64> {
        if let Attribute::U64(v) = self { Some(*v) } else { None }
    }
    pub fn as_bool(&self) -> Option<bool> {
        if let Attribute::Bool(v) = self { Some(*v) } else { None }
    }
    pub fn as_str(&self) -> Option<&str> {
        if let Attribute::Str(v) = self { Some(v.as_str()) } else { None }
    }
    pub fn as_block(&self) -> Option<BlockId> {
        if let Attribute::Block(b) = self { Some(*b) } else { None }
    }

    /// Apply `f` to every [`Attribute::Block`] reachable from this attribute
    /// (including those nested inside [`Attribute::List`]). Non-block variants
    /// are left unchanged.
    ///
    /// Centralizing block-reference walking here means conversion patterns
    /// don't each have to re-implement remapping logic — the
    /// [`ConversionDriver`](crate::conversion::ConversionDriver) calls this
    /// once per op and every dialect inherits the behavior.
    pub fn remap_block_refs<F>(&mut self, f: &mut F)
    where
        F: FnMut(BlockId) -> BlockId,
    {
        match self {
            Attribute::Block(b) => *b = f(*b),
            Attribute::List(items) => {
                for item in items.iter_mut() {
                    item.remap_block_refs(f);
                }
            }
            Attribute::I64(_)
            | Attribute::U64(_)
            | Attribute::Bool(_)
            | Attribute::Str(_) => {}
        }
    }

    /// Visit every [`BlockId`] referenced by this attribute (read-only).
    pub fn for_each_block_ref<F>(&self, f: &mut F)
    where
        F: FnMut(BlockId),
    {
        match self {
            Attribute::Block(b) => f(*b),
            Attribute::List(items) => {
                for item in items.iter() {
                    item.for_each_block_ref(f);
                }
            }
            Attribute::I64(_)
            | Attribute::U64(_)
            | Attribute::Bool(_)
            | Attribute::Str(_) => {}
        }
    }
}

/// Vector of `(key, value)` attribute pairs. Order is preserved so printers
/// produce stable output; lookup is linear which is fine for the low op counts
/// we expect (< 10 attrs per op).
pub type Attributes = Vec<(String, Attribute)>;

/// A single dialect-level SSA op.
#[derive(Debug, Clone)]
pub struct DialectOp {
    /// Arena index inside the owning [`DialectFunction`].
    pub id: OpId,
    /// Dialect-qualified opcode.
    pub op: DialectOpId,
    /// Result SSA values + their types (0..N for N-result ops).
    pub results: Vec<(ValueId, Type)>,
    /// Operand SSA values (positional, in dialect-defined order).
    pub operands: Vec<ValueId>,
    /// Static metadata.
    pub attrs: Attributes,
    /// Optional source-level provenance. Uses the existing tMIR span type via
    /// `llvm2_ir`'s public surface when available; v1 scaffolding leaves this
    /// as an `Option<(u32, u32)>` byte-range placeholder so `llvm2-dialect`
    /// does not need to depend on the full `tmir` crate.
    pub source: Option<SourceRange>,
}

impl DialectOp {
    /// Look up an attribute by key.
    pub fn attr(&self, key: &str) -> Option<&Attribute> {
        self.attrs
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v)
    }

    /// Iterator over result values only (dropping types).
    pub fn result_values(&self) -> impl Iterator<Item = ValueId> + '_ {
        self.results.iter().map(|(v, _)| *v)
    }

    /// Apply `f` to every [`BlockId`] referenced by this op — currently those
    /// live inside [`Attribute::Block`]/nested [`Attribute::List`] values, but
    /// future op variants may grow block-typed operands and should extend this
    /// method rather than special-casing block remapping in every conversion
    /// pattern.
    pub fn remap_block_refs<F>(&mut self, mut f: F)
    where
        F: FnMut(BlockId) -> BlockId,
    {
        for (_k, attr) in self.attrs.iter_mut() {
            attr.remap_block_refs(&mut f);
        }
    }

    /// Read-only visitor for every [`BlockId`] referenced by this op.
    pub fn for_each_block_ref<F>(&self, mut f: F)
    where
        F: FnMut(BlockId),
    {
        for (_k, attr) in self.attrs.iter() {
            attr.for_each_block_ref(&mut f);
        }
    }

    /// Convenience: collect every block referenced by this op into a `Vec`.
    /// Useful for tests and for building CFG summaries.
    pub fn block_refs(&self) -> Vec<BlockId> {
        let mut out = Vec::new();
        self.for_each_block_ref(|b| out.push(b));
        out
    }
}

/// Lightweight `(start, end)` byte range placeholder for provenance.
///
/// Kept decoupled from `tmir::SourceSpan` so this crate does not need to depend
/// on the tMIR crate. A conversion helper can be added later when dialect ops
/// are created directly from tMIR instructions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SourceRange {
    pub start: u32,
    pub end: u32,
}
