// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Conversation sampling strategies.
//!
//! These preserve Python AIPerf's replacement and wraparound semantics from
//! `src/aiperf/dataset/dataset_samplers.py:15-86`, while consuming the native
//! BLAKE3-derived [`aiperf_rng`] streams.

use std::collections::HashMap;
use std::sync::Arc;

use aiperf_rng::{RandomGenerator, RngRoot};

use crate::error::{DatasetError, Result};
use crate::model::{ConversationMetadata, SessionId};

/// Stateful source of authored conversation identifiers.
pub trait Sampler {
    /// Return the next identifier, recycling indefinitely.
    fn next(&mut self) -> SessionId;
}

/// Factory for one named sampler strategy.
pub trait SamplerFactory: Send + Sync {
    /// Stable registration name.
    fn name(&self) -> &str;

    /// Construct fresh sampler state over sampleable conversation metadata.
    fn create(&self, metadata: &[ConversationMetadata], root: RngRoot) -> Result<Box<dyn Sampler>>;
}

/// Factory for uniform random sampling with replacement.
#[derive(Debug, Clone, Copy, Default)]
pub struct RandomSamplerFactory;

impl SamplerFactory for RandomSamplerFactory {
    fn name(&self) -> &str {
        "random"
    }

    fn create(&self, metadata: &[ConversationMetadata], root: RngRoot) -> Result<Box<dyn Sampler>> {
        Ok(Box::new(RandomSampler::from_metadata(metadata, root)?))
    }
}

/// Factory for insertion-order sampling with wraparound.
#[derive(Debug, Clone, Copy, Default)]
pub struct SequentialSamplerFactory;

impl SamplerFactory for SequentialSamplerFactory {
    fn name(&self) -> &str {
        "sequential"
    }

    fn create(
        &self,
        metadata: &[ConversationMetadata],
        _root: RngRoot,
    ) -> Result<Box<dyn Sampler>> {
        Ok(Box::new(SequentialSampler::from_metadata(metadata)?))
    }
}

/// Factory for shuffle-without-replacement sampling.
#[derive(Debug, Clone, Copy, Default)]
pub struct ShuffleSamplerFactory;

impl SamplerFactory for ShuffleSamplerFactory {
    fn name(&self) -> &str {
        "shuffle"
    }

    fn create(&self, metadata: &[ConversationMetadata], root: RngRoot) -> Result<Box<dyn Sampler>> {
        Ok(Box::new(ShuffleSampler::from_metadata(metadata, root)?))
    }
}

/// Extensible name-to-factory registry used to honor loader sampling policy.
#[derive(Clone, Default)]
pub struct SamplerRegistry {
    factories: HashMap<String, Arc<dyn SamplerFactory>>,
}

impl SamplerRegistry {
    /// Create an empty strategy registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register the native random, sequential, and shuffle strategies.
    pub fn with_builtin_strategies() -> Result<Self> {
        let mut registry = Self::new();
        registry.register(RandomSamplerFactory)?;
        registry.register(SequentialSamplerFactory)?;
        registry.register(ShuffleSamplerFactory)?;
        Ok(registry)
    }

    /// Register one factory, rejecting duplicate normalized names.
    pub fn register(&mut self, factory: impl SamplerFactory + 'static) -> Result<()> {
        let name = normalize_name(factory.name());
        if name.is_empty() {
            return Err(DatasetError::Validation(
                "sampler strategy registration name cannot be empty".into(),
            ));
        }
        if self.factories.contains_key(&name) {
            return Err(DatasetError::Validation(format!(
                "duplicate sampler strategy {name:?}"
            )));
        }
        self.factories.insert(name, Arc::new(factory));
        Ok(())
    }

    /// Construct fresh state for a named strategy.
    pub fn create(
        &self,
        name: &str,
        metadata: &[ConversationMetadata],
        root: RngRoot,
    ) -> Result<Box<dyn Sampler>> {
        let name = normalize_name(name);
        self.factories
            .get(&name)
            .ok_or_else(|| {
                let mut available = self.factories.keys().cloned().collect::<Vec<_>>();
                available.sort();
                DatasetError::Validation(format!(
                    "unknown sampler strategy {name:?}; registered strategies: {}",
                    available.join(", ")
                ))
            })?
            .create(metadata, root)
    }
}

fn normalize_name(name: &str) -> String {
    name.trim().to_ascii_lowercase().replace('-', "_")
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

    #[test]
    fn registry_builds_named_strategies_and_rejects_unknown_names() {
        let metadata = ids()
            .into_iter()
            .map(|conversation_id| ConversationMetadata {
                conversation_id,
                turns: Vec::new(),
                context_mode: None,
                dag: None,
            })
            .collect::<Vec<_>>();
        let registry = SamplerRegistry::with_builtin_strategies().unwrap();
        let mut sequential = registry
            .create("SEQUENTIAL", &metadata, RngRoot::new(Some(1)))
            .unwrap();
        assert_eq!(sequential.next().as_str(), "a");
        let mut shuffle = registry
            .create("shuffle", &metadata, RngRoot::new(Some(9)))
            .unwrap();
        let cycle = (0..4).map(|_| shuffle.next()).collect::<HashSet<_>>();
        assert_eq!(cycle.len(), 4);
        assert!(
            registry
                .create("missing", &metadata, RngRoot::new(Some(1)))
                .err()
                .unwrap()
                .to_string()
                .contains("registered strategies")
        );
    }
}
