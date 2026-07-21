// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared transactional registration primitive.
//!
//! Registry batches mutate a clone and replace the target only on success.

use std::collections::BTreeMap;
use std::collections::btree_map::{Keys, Values};
use std::error::Error;
use std::fmt::{self, Display};

use super::registry_id::RegistryId;

/// Apply a mutation atomically, leaving `target` unchanged on error.
pub(crate) fn commit_on_clone<S, E>(
    target: &mut S,
    mutate: impl FnOnce(&mut S) -> Result<(), E>,
) -> Result<(), E>
where
    S: Clone,
{
    let mut staged = target.clone();
    mutate(&mut staged)?;
    *target = staged;
    Ok(())
}

/// A name was registered twice in one registry.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DuplicateName(pub String);

impl Display for DuplicateName {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "duplicate registry name {:?}", self.0)
    }
}

impl Error for DuplicateName {}

/// A name-keyed registry that rejects duplicate names and stages batches atomically.
///
/// Entries iterate in deterministic name order (backed by a [`BTreeMap`]), so
/// descriptor inventories and capability catalogs derived from a registry are
/// stable across builds.
#[derive(Clone, Debug)]
pub struct TransactionalRegistry<T> {
    entries: BTreeMap<RegistryId, T>,
}

impl<T> Default for TransactionalRegistry<T> {
    fn default() -> Self {
        Self {
            entries: BTreeMap::new(),
        }
    }
}

impl<T: Clone> TransactionalRegistry<T> {
    /// Construct an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert one entry, rejecting a name that is already present.
    ///
    /// `name` is a static, code-authored identifier (a descriptor ID or
    /// registration name), never wire input; an empty/invalid spelling is a
    /// build-time bug, not a runtime data error.
    pub fn insert(&mut self, name: impl AsRef<str>, value: T) -> Result<(), DuplicateName> {
        let name = name.as_ref();
        let id = RegistryId::new(name)
            .unwrap_or_else(|error| panic!("registry identifier must be valid: {error}"));
        if self.entries.contains_key(&id) {
            return Err(DuplicateName(name.to_string()));
        }
        self.entries.insert(id, value);
        Ok(())
    }

    /// Apply a multi-entry batch atomically.
    ///
    /// The closure registers against a staged clone; if any step fails, none of
    /// the batch's insertions survive.
    pub fn commit<E>(&mut self, mutate: impl FnOnce(&mut Self) -> Result<(), E>) -> Result<(), E> {
        commit_on_clone(self, mutate)
    }

    /// Borrow a registered entry by name (case/hyphen-insensitive).
    pub fn get(&self, name: &str) -> Option<&T> {
        let id = RegistryId::new(name).ok()?;
        self.entries.get(id.as_str())
    }

    /// Report whether a name is registered (case/hyphen-insensitive).
    pub fn contains(&self, name: &str) -> bool {
        RegistryId::new(name).is_ok_and(|id| self.entries.contains_key(id.as_str()))
    }

    /// Iterate registered names in deterministic order.
    pub fn keys(&self) -> Keys<'_, RegistryId, T> {
        self.entries.keys()
    }

    /// Iterate registered values in deterministic name order.
    pub fn values(&self) -> Values<'_, RegistryId, T> {
        self.entries.values()
    }

    /// Number of registered entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Report whether the registry has no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duplicate_insert_is_rejected_and_leaves_the_original_value() {
        let mut registry = TransactionalRegistry::new();
        registry.insert("x", 1u32).unwrap();
        let error = registry.insert("x", 2u32).unwrap_err();
        assert_eq!(error, DuplicateName("x".to_owned()));
        assert_eq!(registry.get("x"), Some(&1));
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn commit_stages_a_batch_atomically_on_a_later_duplicate() {
        let mut registry = TransactionalRegistry::new();
        registry.insert("a", 1u32).unwrap();

        let result = registry.commit(|staged| {
            staged.insert("b", 2u32)?;
            // The duplicate aborts the batch; the earlier `b` insertion must not
            // remain visible on the committed registry.
            staged.insert("a", 3u32)?;
            Ok::<(), DuplicateName>(())
        });

        assert!(result.is_err());
        assert_eq!(registry.get("a"), Some(&1));
        assert!(!registry.contains("b"));
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn commit_applies_a_whole_batch_on_success() {
        let mut registry = TransactionalRegistry::new();
        registry
            .commit(|staged| {
                staged.insert("a", 1u32)?;
                staged.insert("b", 2u32)?;
                Ok::<(), DuplicateName>(())
            })
            .unwrap();
        assert_eq!(
            registry.keys().map(RegistryId::as_str).collect::<Vec<_>>(),
            ["a", "b"]
        );
        assert_eq!(registry.values().copied().collect::<Vec<_>>(), [1, 2]);
    }
}
