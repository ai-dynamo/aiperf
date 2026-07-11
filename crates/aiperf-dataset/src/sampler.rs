// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Conversation sampling strategies.
//!
//! These preserve Python AIPerf's replacement and wraparound semantics from
//! `src/aiperf/dataset/dataset_samplers.py:15-86`, while consuming the native
//! BLAKE3-derived [`aiperf_rng`] streams.

use aiperf_rng::{RandomGenerator, RngRoot};

use crate::error::{DatasetError, Result};
use crate::model::{ConversationMetadata, SessionId};

/// Stateful source of authored conversation identifiers.
pub trait Sampler {
    /// Return the next identifier, recycling indefinitely.
    fn next(&mut self) -> SessionId;
}

fn validate_ids(ids: Vec<SessionId>) -> Result<Vec<SessionId>> {
    if ids.is_empty() {
        Err(DatasetError::EmptySampler)
    } else {
        Ok(ids)
    }
}

fn root_ids(metadata: &[ConversationMetadata]) -> Vec<SessionId> {
    metadata
        .iter()
        .filter(|conversation| {
            conversation
                .dag
                .as_ref()
                .map(|dag| dag.is_root)
                .unwrap_or(true)
        })
        .map(|conversation| conversation.conversation_id.clone())
        .collect()
}

/// Uniform random sampling with replacement.
pub struct RandomSampler {
    ids: Vec<SessionId>,
    rng: RandomGenerator,
}

impl RandomSampler {
    /// Construct from authored identifiers and a run root seed.
    pub fn new(ids: Vec<SessionId>, root: RngRoot) -> Result<Self> {
        Ok(Self {
            ids: validate_ids(ids)?,
            rng: RandomGenerator::from_seed(root.derive_seed("dataset.sampler.random")),
        })
    }

    /// Construct from sampleable root metadata.
    pub fn from_metadata(metadata: &[ConversationMetadata], root: RngRoot) -> Result<Self> {
        Self::new(root_ids(metadata), root)
    }
}

impl Sampler for RandomSampler {
    fn next(&mut self) -> SessionId {
        self.rng
            .choice(&self.ids)
            .expect("constructor rejects an empty sampler")
            .clone()
    }
}

/// Insertion-order sampling with indefinite wraparound.
pub struct SequentialSampler {
    ids: Vec<SessionId>,
    index: usize,
}

impl SequentialSampler {
    /// Construct from authored identifiers.
    pub fn new(ids: Vec<SessionId>) -> Result<Self> {
        Ok(Self {
            ids: validate_ids(ids)?,
            index: 0,
        })
    }

    /// Construct from sampleable root metadata.
    pub fn from_metadata(metadata: &[ConversationMetadata]) -> Result<Self> {
        Self::new(root_ids(metadata))
    }
}

impl Sampler for SequentialSampler {
    fn next(&mut self) -> SessionId {
        if self.index == self.ids.len() {
            self.index = 0;
        }
        let id = self.ids[self.index].clone();
        self.index += 1;
        id
    }
}

/// Shuffle-without-replacement sampling, reshuffled after every complete cycle.
pub struct ShuffleSampler {
    ids: Vec<SessionId>,
    index: usize,
    rng: RandomGenerator,
}

impl ShuffleSampler {
    /// Construct from authored identifiers and shuffle the first cycle immediately.
    pub fn new(ids: Vec<SessionId>, root: RngRoot) -> Result<Self> {
        let mut ids = validate_ids(ids)?;
        let mut rng = RandomGenerator::from_seed(root.derive_seed("dataset.sampler.shuffle"));
        rng.shuffle(&mut ids);
        Ok(Self { ids, index: 0, rng })
    }

    /// Construct from sampleable root metadata.
    pub fn from_metadata(metadata: &[ConversationMetadata], root: RngRoot) -> Result<Self> {
        Self::new(root_ids(metadata), root)
    }
}

impl Sampler for ShuffleSampler {
    fn next(&mut self) -> SessionId {
        if self.index == self.ids.len() {
            self.rng.shuffle(&mut self.ids);
            self.index = 0;
        }
        let id = self.ids[self.index].clone();
        self.index += 1;
        id
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    fn ids() -> Vec<SessionId> {
        ["a", "b", "c", "d"]
            .into_iter()
            .map(SessionId::from)
            .collect()
    }

    #[test]
    fn empty_samplers_are_rejected() {
        assert!(matches!(
            SequentialSampler::new(Vec::new()),
            Err(DatasetError::EmptySampler)
        ));
    }

    #[test]
    fn sequential_wraps_in_authored_order() {
        let mut sampler = SequentialSampler::new(ids()).unwrap();
        let sampled: Vec<_> = (0..6).map(|_| sampler.next()).collect();
        assert_eq!(
            sampled.iter().map(SessionId::as_str).collect::<Vec<_>>(),
            vec!["a", "b", "c", "d", "a", "b"]
        );
    }

    #[test]
    fn shuffle_visits_every_id_once_per_cycle_and_reproduces() {
        let mut first = ShuffleSampler::new(ids(), RngRoot::new(Some(42))).unwrap();
        let mut second = ShuffleSampler::new(ids(), RngRoot::new(Some(42))).unwrap();
        let a: Vec<_> = (0..8).map(|_| first.next()).collect();
        let b: Vec<_> = (0..8).map(|_| second.next()).collect();
        assert_eq!(a, b);
        assert_eq!(a[..4].iter().collect::<HashSet<_>>().len(), 4);
        assert_eq!(a[4..].iter().collect::<HashSet<_>>().len(), 4);
    }

    #[test]
    fn random_with_replacement_reproduces() {
        let mut first = RandomSampler::new(ids(), RngRoot::new(Some(7))).unwrap();
        let mut second = RandomSampler::new(ids(), RngRoot::new(Some(7))).unwrap();
        let a: Vec<_> = (0..100).map(|_| first.next()).collect();
        let b: Vec<_> = (0..100).map(|_| second.next()).collect();
        assert_eq!(a, b);
    }
}
