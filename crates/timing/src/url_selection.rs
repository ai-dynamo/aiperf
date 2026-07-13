// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Endpoint selection policy for multi-URL runs.
//!
//! The selector
//! deliberately knows nothing about turns or sessions: the issuer advances it
//! on turn zero only, and session state pins that selected index for every
//! continuation turn.

use std::error::Error;
use std::fmt::{Display, Formatter};

/// A pluggable endpoint-index selector.
pub trait UrlSelector {
    /// Return the next endpoint index, which must be strictly less than
    /// [`len`](Self::len).
    fn next_index(&mut self) -> usize;

    /// Number of endpoints addressable by the selector.
    fn len(&self) -> usize;

    /// Whether the selector has no endpoints.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Invalid URL-selector configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UrlSelectionError {
    /// A selector cannot be created without at least one endpoint.
    EmptyUrls,
}

impl Display for UrlSelectionError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyUrls => write!(f, "URL list cannot be empty"),
        }
    }
}

impl Error for UrlSelectionError {}

/// Sequential endpoint selection with wraparound.
pub struct RoundRobinUrlSelector {
    urls: Vec<String>,
    index: usize,
}

impl RoundRobinUrlSelector {
    /// Construct a round-robin selector over a non-empty endpoint list.
    pub fn new(urls: Vec<String>) -> Result<Self, UrlSelectionError> {
        if urls.is_empty() {
            return Err(UrlSelectionError::EmptyUrls);
        }
        Ok(Self { urls, index: 0 })
    }

    /// Endpoints addressed by returned indices, in configured order.
    pub fn urls(&self) -> &[String] {
        &self.urls
    }
}

impl UrlSelector for RoundRobinUrlSelector {
    fn next_index(&mut self) -> usize {
        let current = self.index;
        self.index = (self.index + 1) % self.urls.len();
        current
    }

    fn len(&self) -> usize {
        self.urls.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_an_empty_url_list() {
        assert_eq!(
            RoundRobinUrlSelector::new(Vec::new()).err(),
            Some(UrlSelectionError::EmptyUrls)
        );
    }

    #[test]
    fn one_url_always_selects_zero() {
        let mut selector = RoundRobinUrlSelector::new(vec!["http://one".into()]).unwrap();
        assert_eq!(selector.urls(), ["http://one"]);
        assert_eq!(selector.len(), 1);
        assert!((0..20).all(|_| selector.next_index() == 0));
    }

    #[test]
    fn multiple_urls_cycle_in_order() {
        let urls = vec!["a".into(), "b".into(), "c".into()];
        let mut selector = RoundRobinUrlSelector::new(urls).unwrap();
        let actual: Vec<_> = (0..8).map(|_| selector.next_index()).collect();
        assert_eq!(actual, vec![0, 1, 2, 0, 1, 2, 0, 1]);
    }

    #[test]
    fn trait_is_object_safe() {
        let mut selector: Box<dyn UrlSelector> =
            Box::new(RoundRobinUrlSelector::new(vec!["a".into(), "b".into()]).unwrap());
        assert_eq!(selector.next_index(), 0);
        assert_eq!(selector.next_index(), 1);
    }
}
