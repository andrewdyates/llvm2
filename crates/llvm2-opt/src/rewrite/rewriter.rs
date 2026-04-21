// llvm2-opt - Rewrite actions
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! [`Rewriter`] produces the replacement [`RewriteAction`] once a rule's
//! matcher and constraints have accepted an instruction.

use llvm2_ir::MachInst;

use crate::rewrite::matcher::MatchCtx;

/// What a rule does to the matched instruction.
#[derive(Debug, Clone)]
pub enum RewriteAction {
    /// Keep the instruction unchanged. Used as a safety escape hatch by
    /// rewriters that discover a reason to bail after construction.
    None,
    /// Replace the matched instruction in place.
    Replace(MachInst),
    /// Remove the matched instruction entirely.
    Delete,
}

impl RewriteAction {
    /// Returns true if this action changes the function.
    #[inline]
    pub fn is_change(&self) -> bool {
        !matches!(self, Self::None)
    }
}

/// A rewriter produces a [`RewriteAction`] from a matched context.
pub trait Rewriter: Send + Sync {
    fn apply(&self, ctx: &MatchCtx<'_>) -> RewriteAction;
}

/// Function-pointer rewriter — the ergonomic default for small rules.
///
/// Uses `fn` rather than `Fn` so rewriter values remain `Send + Sync`
/// without boxing trait objects with complex bounds.
pub struct RewriterFn(pub fn(&MatchCtx<'_>) -> RewriteAction);

impl Rewriter for RewriterFn {
    fn apply(&self, ctx: &MatchCtx<'_>) -> RewriteAction {
        (self.0)(ctx)
    }
}
