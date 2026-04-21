// llvm2-opt - Rule struct + builder
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! A [`Rule`] bundles a matcher, a list of constraints, a rewriter, a
//! name, and a benefit score.
//!
//! [`RuleBuilder`] is the ergonomic entry point: call `match_opcode` or
//! `match_category`, chain `.constrain(...)`, set `.benefit(n)`, and
//! finish with `.rewrite_with(...)` or `.rewrite(...)`.

use llvm2_ir::{AArch64Opcode, OpcodeCategory};

use crate::rewrite::constraint::Constraint;
use crate::rewrite::matcher::{CategoryMatcher, MatchCtx, Matcher, OpcodeMatcher};
use crate::rewrite::rewriter::{RewriteAction, Rewriter, RewriterFn};

/// A compiled rewrite rule.
pub struct Rule {
    /// Rule name for diagnostics, logging, and benefit tie-breaking.
    pub name: &'static str,
    /// Benefit score; higher wins when multiple rules match.
    pub benefit: i32,
    /// Matcher: cheap structural check.
    pub matcher: Box<dyn Matcher>,
    /// Constraints: evaluated in order; first failure disqualifies the rule.
    pub constraints: Vec<Box<dyn Constraint>>,
    /// Rewriter: produces the replacement action.
    pub rewriter: Box<dyn Rewriter>,
}

impl Rule {
    /// Evaluate this rule against a context.
    ///
    /// Returns the [`RewriteAction`] if the rule fires, or `None` if the
    /// matcher or any constraint rejects.
    pub fn evaluate(&self, ctx: &MatchCtx<'_>) -> Option<RewriteAction> {
        if !self.matcher.matches(ctx.inst) {
            return None;
        }
        for c in &self.constraints {
            if !c.check(ctx) {
                return None;
            }
        }
        let action = self.rewriter.apply(ctx);
        if action.is_change() {
            Some(action)
        } else {
            None
        }
    }
}

/// Fluent builder for [`Rule`].
pub struct RuleBuilder {
    name: &'static str,
    benefit: i32,
    matcher: Box<dyn Matcher>,
    constraints: Vec<Box<dyn Constraint>>,
}

impl RuleBuilder {
    /// Start a rule that matches any instruction with the given opcode.
    pub fn match_opcode(name: &'static str, opcode: AArch64Opcode) -> Self {
        Self {
            name,
            benefit: 1,
            matcher: Box::new(OpcodeMatcher { opcode }),
            constraints: Vec::new(),
        }
    }

    /// Start a rule that matches any instruction whose opcode falls in
    /// the given category.
    pub fn match_category(name: &'static str, category: OpcodeCategory) -> Self {
        Self {
            name,
            benefit: 1,
            matcher: Box::new(CategoryMatcher { category }),
            constraints: Vec::new(),
        }
    }

    /// Set the benefit score for this rule.
    pub fn benefit(mut self, b: i32) -> Self {
        self.benefit = b;
        self
    }

    /// Add a constraint.
    pub fn constrain<C: Constraint + 'static>(mut self, c: C) -> Self {
        self.constraints.push(Box::new(c));
        self
    }

    /// Finish the rule with a function-pointer rewriter.
    pub fn rewrite_with(self, f: fn(&MatchCtx<'_>) -> RewriteAction) -> Rule {
        Rule {
            name: self.name,
            benefit: self.benefit,
            matcher: self.matcher,
            constraints: self.constraints,
            rewriter: Box::new(RewriterFn(f)),
        }
    }

    /// Finish the rule with an arbitrary boxed rewriter.
    pub fn rewrite(self, r: Box<dyn Rewriter>) -> Rule {
        Rule {
            name: self.name,
            benefit: self.benefit,
            matcher: self.matcher,
            constraints: self.constraints,
            rewriter: r,
        }
    }
}
