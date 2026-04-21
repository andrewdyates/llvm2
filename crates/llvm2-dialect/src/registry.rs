// llvm2-dialect - DialectRegistry
//
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Copyright 2026 Andrew Yates | License: Apache-2.0

//! Registry mapping dialect namespaces to concrete [`Dialect`] impls.
//!
//! Insertion assigns a stable [`DialectId`] that matches the insertion order.

use std::collections::HashMap;

use crate::dialect::{Dialect, OpDef};
use crate::id::{DialectId, DialectOpId};

/// Registry of dialects available to a [`DialectModule`](crate::module::DialectModule).
#[derive(Default)]
pub struct DialectRegistry {
    entries: Vec<Box<dyn Dialect>>,
    by_name: HashMap<String, DialectId>,
}

impl DialectRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a dialect. Returns the assigned [`DialectId`].
    ///
    /// Re-registering a namespace replaces the previous entry but keeps the
    /// original id so existing [`DialectOpId`] handles stay valid.
    pub fn register(&mut self, dialect: Box<dyn Dialect>) -> DialectId {
        let ns = dialect.namespace().to_string();
        if let Some(existing) = self.by_name.get(&ns).copied() {
            self.entries[existing.0 as usize] = dialect;
            return existing;
        }
        let id = DialectId(self.entries.len() as u16);
        self.entries.push(dialect);
        self.by_name.insert(ns, id);
        id
    }

    /// Look up a dialect id by its namespace.
    pub fn by_name(&self, ns: &str) -> Option<DialectId> {
        self.by_name.get(ns).copied()
    }

    /// Dereference a [`DialectId`] to its concrete dialect.
    pub fn get(&self, id: DialectId) -> Option<&dyn Dialect> {
        self.entries.get(id.0 as usize).map(|b| b.as_ref())
    }

    /// Resolve op metadata for a globally-qualified op id.
    pub fn op_def(&self, id: DialectOpId) -> Option<&OpDef> {
        self.get(id.dialect)?.op_def(id.op)
    }

    /// Total number of registered dialects.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterate over `(DialectId, &dyn Dialect)` pairs in insertion order.
    pub fn iter(&self) -> impl Iterator<Item = (DialectId, &dyn Dialect)> {
        self.entries
            .iter()
            .enumerate()
            .map(|(i, d)| (DialectId(i as u16), d.as_ref()))
    }
}

impl std::fmt::Debug for DialectRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut dbg = f.debug_struct("DialectRegistry");
        for (id, d) in self.iter() {
            dbg.field(&format!("{:?}", id), &d.namespace());
        }
        dbg.finish()
    }
}
