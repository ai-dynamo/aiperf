// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Conversation sampling strategies.
//!
//! Sampling preserves replacement and wraparound semantics while consuming
//! BLAKE3-derived [`crate::rng`] streams.

use std::collections::HashMap;
use std::sync::Arc;

use crate::cellular::partition::{CellPartition, ModuloCellPartition};
use crate::extensions::RegistryId;
use crate::rng::{ConfiguredRandomGenerator, RandomGenerator, RngRoot};

use crate::dataset::error::{DatasetError, Result};
use crate::dataset::model::{ConversationMetadata, SessionId};

/// Stateful source of authored conversation identifiers.
pub trait Sampler {
    /// Return the next identifier, recycling indefinitely.
    fn next(&mut self) -> SessionId;

    /// Advance the sampler's internal state by one draw without materializing the
    /// identifier. The default recycles through [`Self::next`] and discards the
    /// clone; samplers whose draw clones an owned [`SessionId`] override this to
    /// skip that allocation. [`PartitionedSampler`] uses it to consume the draws
    /// this cell does not own without heap-allocating an id it immediately drops.
    ///
    /// The advance MUST be state-identical to one [`Self::next`] call so
    /// partitioned draw ordering stays byte-exact.
    fn advance(&mut self) {
        let _ = self.next();
    }
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
    factories: HashMap<RegistryId, Arc<dyn SamplerFactory>>,
}

impl SamplerRegistry {
    /// Create an empty strategy registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register the native random, sequential, and shuffle strategies.
    pub fn with_builtin_strategies() -> Result<Self> {
        let mut registry = Self::new();
        registry.register_builtin_strategies()?;
        Ok(registry)
    }

    /// Register the native random, sequential, and shuffle strategies into an
    /// existing registry. Shared by [`Self::with_builtin_strategies`] and the
    /// built-in sampler `AIPerfExtension` so both compose the identical set.
    pub fn register_builtin_strategies(&mut self) -> Result<()> {
        self.register(RandomSamplerFactory)?;
        self.register(SequentialSamplerFactory)?;
        self.register(ShuffleSamplerFactory)?;
        Ok(())
    }

    /// Register one factory, rejecting duplicate normalized names.
    pub fn register(&mut self, factory: impl SamplerFactory + 'static) -> Result<()> {
        let id = RegistryId::new(factory.name()).map_err(|_| {
            DatasetError::Validation("sampler strategy registration name cannot be empty".into())
        })?;
        if self.factories.contains_key(&id) {
            return Err(DatasetError::Validation(format!(
                "duplicate sampler strategy {:?}",
                id.as_str()
            )));
        }
        self.factories.insert(id, Arc::new(factory));
        Ok(())
    }

    /// Construct fresh state for a named strategy.
    pub fn create(
        &self,
        name: &str,
        metadata: &[ConversationMetadata],
        root: RngRoot,
    ) -> Result<Box<dyn Sampler>> {
        let id = RegistryId::new(name)
            .map_err(|_| DatasetError::Validation(format!("unknown sampler strategy {name:?}")))?;
        self.factories
            .get(id.as_str())
            .ok_or_else(|| {
                let mut available = self
                    .factories
                    .keys()
                    .map(RegistryId::as_str)
                    .collect::<Vec<_>>();
                available.sort();
                DatasetError::Validation(format!(
                    "unknown sampler strategy {name:?}; registered strategies: {}",
                    available.join(", ")
                ))
            })?
            .create(metadata, root)
    }
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
    rng: ConfiguredRandomGenerator,
}

impl RandomSampler {
    /// Construct from authored identifiers and a run root seed.
    pub fn new(ids: Vec<SessionId>, root: RngRoot) -> Result<Self> {
        Ok(Self {
            ids: validate_ids(ids)?,
            rng: root.derive_generator("dataset.sampler.random"),
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

    fn advance(&mut self) {
        // Mirror `next`'s single rng draw exactly, skipping only the clone.
        let _ = self.rng.choice(&self.ids);
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

    fn advance(&mut self) {
        if self.index == self.ids.len() {
            self.index = 0;
        }
        self.index += 1;
    }
}

/// Shuffle-without-replacement sampling, reshuffled after every complete cycle.
pub struct ShuffleSampler {
    ids: Vec<SessionId>,
    index: usize,
    rng: ConfiguredRandomGenerator,
}

impl ShuffleSampler {
    /// Construct from authored identifiers and shuffle the first cycle immediately.
    pub fn new(ids: Vec<SessionId>, root: RngRoot) -> Result<Self> {
        let mut ids = validate_ids(ids)?;
        let mut rng: ConfiguredRandomGenerator = root.derive_generator("dataset.sampler.shuffle");
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

    fn advance(&mut self) {
        if self.index == self.ids.len() {
            self.rng.shuffle(&mut self.ids);
            self.index = 0;
        }
        self.index += 1;
    }
}

/// Wraps a sampler so this cell yields only the positions it owns.
///
/// Over a sequential inner sampler the draw position is the instance index, so cell
/// `k` yields instances `{k, k+N, k+2N, …}`; paired with the autonomous issuer's
/// `global_ordinal = position`, a merged `N`-cell run lands each instance at the
/// same ordinal a single cell would — a byte-identical trace set. (Over a random
/// inner sampler it partitions draw order rather than instance index; a merge stays
/// well-formed but is not one-cell-identical.)
pub struct PartitionedSampler {
    inner: Box<dyn Sampler>,
    partition: ModuloCellPartition,
    position: u64,
}

impl PartitionedSampler {
    /// Wraps `inner` with `partition`'s ownership filter.
    pub fn new(inner: Box<dyn Sampler>, partition: ModuloCellPartition) -> Self {
        Self {
            inner,
            partition,
            position: 0,
        }
    }

    /// Wraps `inner` with this process's cell partition when it is one cell of a
    /// multi-cell run; otherwise returns `inner` unchanged (single process or the
    /// identity partition), so non-cell sampling stays byte-identical.
    pub fn from_env(inner: Box<dyn Sampler>) -> Box<dyn Sampler> {
        Self::for_partition(inner, ModuloCellPartition::from_env())
    }

    /// A multi-cell partition (`cell_count > 1`) applies the ownership filter;
    /// `None` or the identity partition returns `inner` unchanged. Unlike
    /// [`Self::from_env`], this accepts worker-local partitions that process-global
    /// environment variables cannot represent.
    pub fn for_partition(
        inner: Box<dyn Sampler>,
        partition: Option<ModuloCellPartition>,
    ) -> Box<dyn Sampler> {
        match partition {
            Some(partition) if partition.cell_count() > 1 => Box::new(Self::new(inner, partition)),
            _ => inner,
        }
    }
}

impl Sampler for PartitionedSampler {
    fn next(&mut self) -> SessionId {
        loop {
            let owned = self.partition.owns(self.position);
            self.position += 1;
            if owned {
                return self.inner.next();
            }
            // Skip the unowned draw without cloning the id it would yield.
            self.inner.advance();
        }
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

    /// The whole point of `at_position`: it must agree, element for element,
    /// with walking the sampler statefully — including across recycle wraps.
    /// That agreement is what lets W threads each compute their own slice of
    /// one draw sequence with no shared state.
    #[test]
    fn at_position_agrees_with_walking_the_sequential_sampler() {
        let mut walked = SequentialSampler::new(ids()).unwrap();
        let addressed = SequentialSampler::new(ids()).unwrap();
        // `ids()` has 4 entries; 10 draws exercises two full wraps.
        for position in 0..10u64 {
            assert_eq!(
                addressed.at_position(position),
                Some(walked.next()),
                "position {position} must equal the {position}th stateful draw"
            );
        }
    }

    /// RNG-stateful strategies have no closed form, and must say so rather than
    /// guess: the caller falls back to the stateful draw on `None`.
    #[test]
    fn rng_stateful_samplers_report_no_position_addressing() {
        let root = RngRoot::new(Some(7));
        assert!(
            RandomSampler::new(ids(), root)
                .unwrap()
                .at_position(0)
                .is_none()
        );
        assert!(
            ShuffleSampler::new(ids(), root)
                .unwrap()
                .at_position(0)
                .is_none()
        );
    }

    #[test]
    fn partitioned_sampler_yields_disjoint_owned_positions() {
        // Over a sequential inner (position == instance index), the two cells of a
        // 2-cell run split the instance space: cell 0 owns even positions, cell 1 odd.
        let mut cell0 = PartitionedSampler::new(
            Box::new(SequentialSampler::new(ids()).unwrap()),
            ModuloCellPartition::new(0, 2).unwrap(),
        );
        let mut cell1 = PartitionedSampler::new(
            Box::new(SequentialSampler::new(ids()).unwrap()),
            ModuloCellPartition::new(1, 2).unwrap(),
        );
        // positions 0,2 -> a,c ; positions 1,3 -> b,d ; union is the whole set.
        assert_eq!(
            [cell0.next(), cell0.next()]
                .iter()
                .map(SessionId::as_str)
                .collect::<Vec<_>>(),
            vec!["a", "c"]
        );
        assert_eq!(
            [cell1.next(), cell1.next()]
                .iter()
                .map(SessionId::as_str)
                .collect::<Vec<_>>(),
            vec!["b", "d"]
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
                accuracy: None,
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
