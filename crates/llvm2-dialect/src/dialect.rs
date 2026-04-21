// llvm2-dialect - Dialect trait + OpDef + Capabilities
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Dropbox, Inc. | License: Apache-2.0

//! [`Dialect`] trait, [`OpDef`] descriptors, and capability bitflags.
//!
//! A dialect is a bundle of related ops identified by a stable string
//! namespace. An [`OpDef`] describes a single op's name, capabilities,
//! expected arity, and a per-position [`TypeConstraint`] vocabulary that
//! declarative rewrite engines (#393) can match on.
//!
//! Capabilities are a static flag set (v1 stand-in for MLIR-style op
//! interfaces; adequate for the boolean queries passes currently need — see
//! the design doc §3.3).
//!
//! `TypeConstraint` follows `designs/2026-04-18-aggregates-dialects-coordination.md`
//! §5: it references `llvm2_ir::function::Type` directly rather than defining
//! a parallel type enum. This is the shared-substrate invariant — dialects
//! layer on top of the value-type system, not beside it.

use llvm2_ir::Type;

use crate::id::OpCode;

/// Per-position type constraint used by [`OpDef::operand_types`] and
/// [`OpDef::result_types`].
///
/// This vocabulary mirrors §5 of `designs/2026-04-18-aggregates-dialects-coordination.md`.
/// Intended consumers:
///   * `validate_type_constraints` — runtime legality checking at
///     pass boundaries.
///   * The declarative rewrite engine (#393) — pattern matching against
///     operand types without string parsing.
///
/// Constraint positions refer to the op's `operand_types` / `result_types`
/// arrays in definition order. `SameAs(i)` refers to `operand_types[i]`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TypeConstraint {
    /// Any type is accepted (wildcard). Use sparingly — mainly for ops whose
    /// polymorphism cannot be described by the other variants.
    Any,
    /// Any integer scalar (I8..I128).
    AnyInt,
    /// Any floating-point scalar (F32/F64).
    AnyFloat,
    /// Any scalar type (non-aggregate).
    AnyScalar,
    /// Any aggregate type (Struct/Array — see `Type::is_aggregate`).
    Aggregate,
    /// Exact match against a specific `llvm2_ir::Type`.
    ///
    /// Spelled `Specific` (rather than `ExactScalar`) because the coord spec's
    /// constraint set must express "exactly `Struct([I64, Ptr])`" as well as
    /// "exactly `I32`" — there is no scalar-only restriction.
    Specific(Type),
    /// Must match the type at `operand_types[idx]`. Used to encode
    /// polymorphic ops like `tmir.add` whose operand types must agree.
    SameAs(usize),
}

impl TypeConstraint {
    /// Check whether `ty` satisfies this constraint, given the current op's
    /// operand type list `operand_tys` (used to resolve `SameAs`).
    ///
    /// `operand_tys` is positional: entry `i` is the resolved type of the op's
    /// `i`-th operand, or `None` if that operand's type could not be resolved
    /// (e.g. the caller supplied a partial value-type env to
    /// [`validate_type_constraints_with_env`](crate::pass::validate_type_constraints_with_env)).
    /// Preserving positions is load-bearing: `SameAs(idx)` is a position-
    /// relative reference, and collapsing unresolved slots out of the list
    /// would silently shift later indices (see issue #410).
    ///
    /// `SameAs` on result positions resolves against operand types, which is
    /// the only reference the coord spec contemplates. If the index is out of
    /// range, or the referenced slot is `None` (unresolved), the constraint
    /// fails closed (returns `false`) rather than panicking or silently
    /// resolving against a different slot — callers typically surface a
    /// legality violation for that.
    pub fn accepts(&self, ty: &Type, operand_tys: &[Option<Type>]) -> bool {
        match self {
            TypeConstraint::Any => true,
            TypeConstraint::AnyInt => ty.is_int(),
            TypeConstraint::AnyFloat => ty.is_float(),
            TypeConstraint::AnyScalar => ty.is_scalar(),
            TypeConstraint::Aggregate => ty.is_aggregate(),
            TypeConstraint::Specific(expected) => ty == expected,
            TypeConstraint::SameAs(idx) => operand_tys
                .get(*idx)
                .and_then(|slot| slot.as_ref())
                .map(|reference| reference == ty)
                .unwrap_or(false),
        }
    }
}

/// Static capability bitflags attached to each [`OpDef`].
///
/// Hand-rolled to avoid a `bitflags` crate dependency. All constructors use
/// `from_bits_truncate` semantics (unknown bits are ignored on `from_bits`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Capabilities(pub u32);

impl Capabilities {
    pub const EMPTY: Self = Self(0);
    pub const PURE: Self = Self(1 << 0);
    pub const HAS_PARALLELISM: Self = Self(1 << 1);
    pub const IS_REDUCTION: Self = Self(1 << 2);
    pub const IS_FOLD: Self = Self(1 << 3);
    pub const IS_MAP: Self = Self(1 << 4);
    pub const BOUNDED_LOOPS: Self = Self(1 << 5);
    pub const HAS_SIDE_EFFECT: Self = Self(1 << 6);
    pub const IS_TERMINATOR: Self = Self(1 << 7);

    /// Compose multiple capability flags.
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Returns true if every flag in `other` is present in `self`.
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    pub const fn bits(self) -> u32 {
        self.0
    }

    pub const fn is_pure(self) -> bool {
        self.contains(Self::PURE)
    }

    pub const fn has_side_effect(self) -> bool {
        self.contains(Self::HAS_SIDE_EFFECT)
    }

    pub const fn is_terminator(self) -> bool {
        self.contains(Self::IS_TERMINATOR)
    }
}

impl std::ops::BitOr for Capabilities {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        self.union(rhs)
    }
}

/// Expected operand or result count for an op.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Arity {
    /// Exactly N values.
    Fixed(u16),
    /// Zero-or-more variadic. `None` = any; `Some(max)` = at most `max`.
    Variadic(Option<u16>),
}

impl Arity {
    /// Check whether a runtime count matches this arity constraint.
    pub fn accepts(self, n: usize) -> bool {
        match self {
            Arity::Fixed(k) => n == k as usize,
            Arity::Variadic(None) => true,
            Arity::Variadic(Some(max)) => n <= max as usize,
        }
    }
}

/// Metadata for a single op within a [`Dialect`].
///
/// `operand_types` / `result_types` carry the per-position
/// [`TypeConstraint`] vocabulary required by
/// `designs/2026-04-18-aggregates-dialects-coordination.md` §5. For
/// fixed-arity ops the slice length should equal the arity; for
/// variadic ops (`Arity::Variadic`) a single-element slice applies to
/// every operand/result (see [`validate_type_constraints`](crate::pass::validate_type_constraints)
/// for the matching rules).
///
/// Ops that have no type expectations (legacy callers, rapid prototyping)
/// can leave these as `&[]`; type validation is skipped when the constraint
/// slice is empty. New ops should always populate them — silent drift from
/// the coord spec is exactly the bug this field closes.
#[derive(Debug, Clone)]
pub struct OpDef {
    pub op: OpCode,
    pub name: &'static str,
    pub capabilities: Capabilities,
    pub num_operands: Arity,
    pub num_results: Arity,
    /// Operand type constraints in positional order. See type-level docs for
    /// fixed-vs-variadic matching rules.
    pub operand_types: &'static [TypeConstraint],
    /// Result type constraints in positional order.
    pub result_types: &'static [TypeConstraint],
}

/// A dialect is a namespaced bundle of ops.
///
/// Implementations are plugged into a [`DialectRegistry`](crate::registry::DialectRegistry).
pub trait Dialect: Send + Sync {
    /// Stable string namespace (e.g. `"verif"`, `"tmir"`, `"machir"`).
    fn namespace(&self) -> &'static str;

    /// Enumerate every op defined by this dialect.
    fn ops(&self) -> &[OpDef];

    /// Look up a single op by opcode. Default implementation linearly scans
    /// [`Dialect::ops`]; dialects with many ops may override with a faster
    /// lookup.
    fn op_def(&self, op: OpCode) -> Option<&OpDef> {
        self.ops().iter().find(|d| d.op == op)
    }
}
