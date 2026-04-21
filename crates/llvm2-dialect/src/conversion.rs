// llvm2-dialect - ConversionPattern + Rewriter + Driver
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! One-op-in / N-ops-out conversion patterns and the greedy [`ConversionDriver`]
//! that runs them over a module.
//!
//! This is an intentionally small subset of MLIR's dialect-conversion
//! framework. The goal is to demonstrate progressive lowering end-to-end;
//! partial conversion, type converters, and materializations are future work
//! (see design doc §10).

use std::collections::HashMap;

use llvm2_ir::Type;

use crate::id::{BlockId, DialectOpId, OpId, ValueId};
use crate::module::{DialectFunction, DialectModule};
use crate::op::{Attributes, DialectOp, SourceRange};

/// Error returned by conversion logic.
#[derive(Debug, Clone)]
pub enum ConversionError {
    /// Pattern matched but failed to produce replacement ops.
    RewriteFailed(String),
    /// No pattern is registered for this source op.
    NoPatternForOp(DialectOpId),
    /// The rewriter was asked to emit an unknown op.
    EmitFailed(String),
}

impl std::fmt::Display for ConversionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConversionError::RewriteFailed(m) => write!(f, "rewrite failed: {}", m),
            ConversionError::NoPatternForOp(id) => write!(f, "no pattern for op {:?}", id),
            ConversionError::EmitFailed(m) => write!(f, "emit failed: {}", m),
        }
    }
}

impl std::error::Error for ConversionError {}

/// A rewriter sees a single source op and emits zero or more replacement ops
/// into `rewriter`. Replacement ops must be in the destination dialect.
pub trait ConversionPattern {
    /// The op this pattern replaces.
    fn source_op(&self) -> DialectOpId;

    /// Produce replacements. The rewriter is scoped to the current block — the
    /// driver appends emitted ops in sequence into a fresh "lowered" function
    /// and records the mapping from old result values to new ones.
    fn rewrite(
        &self,
        op: &DialectOp,
        rewriter: &mut Rewriter<'_>,
    ) -> Result<(), ConversionError>;
}

/// Scratch buffer passed to [`ConversionPattern::rewrite`]. Accumulates new
/// ops and remembers how the source op's result `ValueId`s map to freshly
/// allocated destination `ValueId`s.
///
/// The rewriter also exposes the source→destination `BlockId` map so that
/// control-flow patterns can translate successor references via
/// [`Rewriter::map_block`].
pub struct Rewriter<'a> {
    /// The function being built.
    function: &'a mut DialectFunction,
    /// The destination block where new ops are appended. This points at the
    /// destination block **matching the source block** that contains the op
    /// currently being rewritten — the driver advances it as it walks blocks.
    block: BlockId,
    /// Mapping from source-function ValueIds to destination-function ValueIds.
    /// Pattern implementations use [`Rewriter::map`] to translate operands and
    /// [`Rewriter::bind_result`] to declare replacements for source-op results.
    value_map: &'a mut HashMap<ValueId, ValueId>,
    /// Mapping from source-function BlockIds to destination-function BlockIds.
    /// Shared across the full walk so patterns can reference *any* destination
    /// block, not just the current one (e.g. a branch in block 0 that targets
    /// block 2).
    block_map: &'a HashMap<BlockId, BlockId>,
}

impl<'a> Rewriter<'a> {
    /// Translate a source-function `ValueId` into its destination-function
    /// replacement. Returns `None` if the value has not been mapped yet (e.g.
    /// pattern was applied before the op defining that value was converted —
    /// caller should skip / retry).
    pub fn map(&self, src: ValueId) -> Option<ValueId> {
        self.value_map.get(&src).copied()
    }

    /// Translate a source-function `BlockId` into its destination-function
    /// replacement. The driver pre-populates the block map with one entry per
    /// source block *before* running any patterns, so this is guaranteed to
    /// return `Some` for every block present in the source function.
    pub fn map_block(&self, src: BlockId) -> Option<BlockId> {
        self.block_map.get(&src).copied()
    }

    /// Allocate a fresh destination-function `ValueId`.
    pub fn alloc_value(&mut self) -> ValueId {
        self.function.alloc_value()
    }

    /// The destination block currently being filled.
    pub fn current_block(&self) -> BlockId {
        self.block
    }

    /// Declare that a source-function `src` result is replaced by destination
    /// `dst` in the output. Operands referring to `src` in later source ops
    /// will be translated to `dst` automatically.
    pub fn bind_result(&mut self, src: ValueId, dst: ValueId) {
        self.value_map.insert(src, dst);
    }

    /// Append an op into the current destination block. Any `Attribute::Block`
    /// references inside `attrs` are left as-is — callers that emit control
    /// flow should translate those through [`Rewriter::map_block`] before
    /// calling `emit` so the op refers to destination blocks, not stale source
    /// blocks.
    pub fn emit(
        &mut self,
        op: DialectOpId,
        results: Vec<(ValueId, Type)>,
        operands: Vec<ValueId>,
        attrs: Attributes,
        source: Option<SourceRange>,
    ) -> OpId {
        self.function.append_op(self.block, op, results, operands, attrs, source)
    }
}

/// Greedy driver: converts every function in a module in a single pass, using
/// the registered patterns indexed by source [`DialectOpId`].
pub struct ConversionDriver {
    patterns: HashMap<DialectOpId, Box<dyn ConversionPattern>>,
}

impl Default for ConversionDriver {
    fn default() -> Self {
        Self { patterns: HashMap::new() }
    }
}

impl ConversionDriver {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, pattern: Box<dyn ConversionPattern>) {
        let key = pattern.source_op();
        self.patterns.insert(key, pattern);
    }

    /// Run all registered patterns over every function in `module`. Each
    /// function is rebuilt: a fresh [`DialectFunction`] is constructed, ops
    /// are streamed through their matching pattern in original program order,
    /// and the output replaces the input.
    ///
    /// Ops whose `DialectOpId` matches no registered pattern are copied
    /// through verbatim — this lets a single driver handle mixed-dialect
    /// input/output (e.g. keeping `machir.*` ops unchanged while lowering
    /// `tmir.*`).
    pub fn run(&self, module: &mut DialectModule) -> Result<(), ConversionError> {
        for idx in 0..module.functions.len() {
            let rebuilt = self.run_on_function(&module.functions[idx])?;
            module.functions[idx] = rebuilt;
        }
        Ok(())
    }

    fn run_on_function(
        &self,
        src: &DialectFunction,
    ) -> Result<DialectFunction, ConversionError> {
        // --- 1. Build destination skeleton ---
        //
        // Start with a fresh function with the same signature. The entry
        // block is created by `DialectFunction::new` (at index 0) along with
        // function-level parameters; any *additional* source blocks are
        // re-created one-for-one with their block params re-allocated.
        let param_types: Vec<Type> = src.params.iter().map(|(_, t)| t.clone()).collect();
        let mut dst = DialectFunction::new(src.name.clone(), param_types, src.results.clone());

        let mut value_map: HashMap<ValueId, ValueId> = HashMap::new();
        let mut block_map: HashMap<BlockId, BlockId> = HashMap::new();

        // Map function params (entry-block params, conceptually).
        for (i, (src_v, _)) in src.params.iter().enumerate() {
            let dst_v = dst.params[i].0;
            value_map.insert(*src_v, dst_v);
        }

        // Mirror each source block in the destination, preserving block order
        // and block parameters. The first source block reuses the destination
        // entry block allocated by `DialectFunction::new`; any block params on
        // the entry block are allocated and appended onto the destination
        // entry block directly so the one-to-one ordering is preserved.
        debug_assert!(
            !src.blocks.is_empty(),
            "source function must have at least one block"
        );
        for (i, src_block) in src.blocks.iter().enumerate() {
            let dst_block_id = if i == 0 {
                // Entry block: reuse the one created by `DialectFunction::new`.
                // If the source entry block declares block params (unusual —
                // entry-block params normally are function params), mirror
                // them onto the destination entry block so the bookkeeping
                // matches the rest of the code path.
                let entry = dst
                    .entry_block()
                    .expect("fresh function has entry block");
                for (src_v, ty) in &src_block.params {
                    let v = dst.alloc_value();
                    dst.blocks[entry.0 as usize]
                        .params
                        .push((v, ty.clone()));
                    value_map.insert(*src_v, v);
                }
                entry
            } else {
                let param_tys: Vec<Type> =
                    src_block.params.iter().map(|(_, t)| t.clone()).collect();
                let (new_block, new_param_vs) = dst.new_block_with_params(param_tys);
                for ((src_v, _), dst_v) in
                    src_block.params.iter().zip(new_param_vs.iter())
                {
                    value_map.insert(*src_v, *dst_v);
                }
                new_block
            };
            block_map.insert(src_block.id, dst_block_id);
        }

        // --- 2. Walk blocks in order, converting / copying ops ---
        //
        // Iterating by `src.blocks` preserves source program order and keeps
        // control flow ops (branches, returns) physically inside the block
        // where they were written. Previously, `run_on_function` collapsed
        // every op into a single entry block which silently scrambled
        // multi-block functions — that is the F2 soundness bug this rewrite
        // fixes.
        for src_block in src.blocks.iter() {
            let dst_block = *block_map
                .get(&src_block.id)
                .expect("all source blocks were pre-mapped");
            for op_id in &src_block.ops {
                let src_op = &src.ops[op_id.0 as usize];
                if let Some(pattern) = self.patterns.get(&src_op.op) {
                    // Translate operands AND block-ref attrs upfront so the
                    // pattern always sees destination-space identifiers.
                    let translated_op =
                        self.translate_op(src_op, &value_map, &block_map);
                    let mut rewriter = Rewriter {
                        function: &mut dst,
                        block: dst_block,
                        value_map: &mut value_map,
                        block_map: &block_map,
                    };
                    pattern.rewrite(&translated_op, &mut rewriter)?;
                } else {
                    // Copy-through: translate operands, reallocate result
                    // ValueIds, remap BlockId attrs, and append verbatim.
                    let operands: Vec<ValueId> = src_op
                        .operands
                        .iter()
                        .map(|v| value_map.get(v).copied().unwrap_or(*v))
                        .collect();
                    let results: Vec<(ValueId, Type)> = src_op
                        .results
                        .iter()
                        .map(|(v, t)| {
                            let fresh = dst.alloc_value();
                            value_map.insert(*v, fresh);
                            (fresh, t.clone())
                        })
                        .collect();
                    let mut attrs = src_op.attrs.clone();
                    // Remap every block reference through the block map.
                    // Unknown blocks (should not happen) are left as-is.
                    for (_k, attr) in attrs.iter_mut() {
                        attr.remap_block_refs(&mut |b| {
                            block_map.get(&b).copied().unwrap_or(b)
                        });
                    }
                    dst.append_op(
                        dst_block,
                        src_op.op,
                        results,
                        operands,
                        attrs,
                        src_op.source,
                    );
                }
            }
        }

        Ok(dst)
    }

    /// Build a shallow copy of `src_op` with its operands remapped through
    /// `value_map` **and** any `Attribute::Block` references remapped through
    /// `block_map`. Results are not reallocated here — patterns decide which
    /// destination values to bind via [`Rewriter::bind_result`].
    fn translate_op(
        &self,
        src_op: &DialectOp,
        value_map: &HashMap<ValueId, ValueId>,
        block_map: &HashMap<BlockId, BlockId>,
    ) -> DialectOp {
        let operands: Vec<ValueId> = src_op
            .operands
            .iter()
            .map(|v| value_map.get(v).copied().unwrap_or(*v))
            .collect();
        let mut attrs = src_op.attrs.clone();
        for (_k, attr) in attrs.iter_mut() {
            attr.remap_block_refs(&mut |b| block_map.get(&b).copied().unwrap_or(b));
        }
        DialectOp {
            id: src_op.id,
            op: src_op.op,
            results: src_op.results.clone(),
            operands,
            attrs,
            source: src_op.source,
        }
    }
}
