// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Bounded local admission of scored NativeGraph episode matrices.

use std::{
    cell::RefCell,
    collections::{BTreeMap, VecDeque},
    fmt,
    rc::Rc,
};

use async_trait::async_trait;
use futures::{stream::FuturesUnordered, stream::StreamExt};
use tokio::sync::watch;

use crate::eval::{
    ArtifactDigest, AttemptId, EpisodeResult, HarborTaskPackage, ModelCapacityKey,
    ResolvedEpisodeTrial, ResolvedNativeGraphSuite, ResourceLeaseRequest,
};

/// Fixed resource capacities shared by one local matrix scheduler instance.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResourceLimits {
    episode_slots: usize,
    cpu_units: u32,
    memory_bytes: u64,
    model_binding_units: BTreeMap<ModelCapacityKey, u32>,
}

impl ResourceLimits {
    /// Creates finite positive global capacities for local episode admission.
    pub fn new(
        episode_slots: usize,
        cpu_units: u32,
        memory_bytes: u64,
        model_binding_units: BTreeMap<ModelCapacityKey, u32>,
    ) -> Result<Self, MatrixError> {
        if episode_slots == 0 {
            return Err(MatrixError::ZeroCapacity("episode_slots"));
        }
        if cpu_units == 0 {
            return Err(MatrixError::ZeroCapacity("cpu_units"));
        }
        if memory_bytes == 0 {
            return Err(MatrixError::ZeroCapacity("memory_bytes"));
        }
        if model_binding_units.values().any(|units| *units == 0) {
            return Err(MatrixError::ZeroCapacity("model_binding_units"));
        }
        Ok(Self {
            episode_slots,
            cpu_units,
            memory_bytes,
            model_binding_units,
        })
    }

    /// Returns the maximum concurrently admitted episodes.
    pub const fn episode_slots(&self) -> usize {
        self.episode_slots
    }

    /// Returns the global CPU admission capacity.
    pub const fn cpu_units(&self) -> u32 {
        self.cpu_units
    }

    /// Returns the global memory admission capacity in bytes.
    pub const fn memory_bytes(&self) -> u64 {
        self.memory_bytes
    }

    /// Returns package-scoped per-model capacities in deterministic key order.
    pub fn model_binding_units(&self) -> &BTreeMap<ModelCapacityKey, u32> {
        &self.model_binding_units
    }
}

/// One trial assigned to a stable local matrix output slot.
#[derive(Clone, Debug, PartialEq)]
pub struct EpisodeAssignment {
    output_index: usize,
    trial: ResolvedEpisodeTrial,
}

impl EpisodeAssignment {
    fn new(output_index: usize, trial: ResolvedEpisodeTrial) -> Self {
        Self {
            output_index,
            trial,
        }
    }

    /// Returns this attempt's stable manifest output position.
    pub const fn output_index(&self) -> usize {
        self.output_index
    }

    /// Returns this attempt's authored trial-axis position.
    pub const fn manifest_index(&self) -> usize {
        self.trial.manifest_index()
    }

    /// Borrows the expanded immutable trial.
    pub fn trial(&self) -> &ResolvedEpisodeTrial {
        &self.trial
    }

    /// Borrows the immutable trial identity expected in the returned result.
    pub fn trial_digest(&self) -> &ArtifactDigest {
        self.trial.trial_digest()
    }

    /// Borrows this attempt's deterministic identity expected in the returned result.
    pub fn attempt_id(&self) -> &AttemptId {
        self.trial.attempt_id()
    }

    /// Borrows the importer-owned package snapshot for this assigned episode.
    pub fn package(&self) -> &HarborTaskPackage {
        self.trial.package()
    }

    /// Borrows the bounded resources admitted for this episode.
    pub fn resources(&self) -> &ResourceLeaseRequest {
        self.trial.resources()
    }
}

/// Runs one fully assigned episode through its execution and evaluation boundary.
#[async_trait(?Send)]
pub trait EpisodeRunner {
    /// Executes exactly one admitted assignment and returns its immutable result facts.
    async fn run(&self, assignment: EpisodeAssignment) -> Result<EpisodeResult, MatrixError>;
}

/// Executes a resolved suite without changing its manifest ordering.
#[async_trait(?Send)]
pub trait NativeGraphSuiteScheduler {
    /// Runs all assigned trials through the supplied episode boundary.
    async fn run(
        &self,
        suite: ResolvedNativeGraphSuite,
        runner: Rc<dyn EpisodeRunner>,
    ) -> Result<Vec<EpisodeResult>, MatrixError>;
}

/// Creates one local scheduler after the application freezes its selected capability.
pub trait SuiteSchedulerFactory: Send + Sync {
    /// Creates a scheduler that owns only the supplied bounded local resources.
    fn create(
        &self,
        limits: ResourceLimits,
    ) -> Result<Rc<dyn NativeGraphSuiteScheduler>, MatrixError>;
}

/// Factory for the native bounded local matrix scheduler.
#[derive(Clone, Copy, Debug, Default)]
pub struct LocalNativeGraphSuiteSchedulerFactory;

impl SuiteSchedulerFactory for LocalNativeGraphSuiteSchedulerFactory {
    fn create(
        &self,
        limits: ResourceLimits,
    ) -> Result<Rc<dyn NativeGraphSuiteScheduler>, MatrixError> {
        Ok(Rc::new(LocalNativeGraphSuiteScheduler::new(limits)?))
    }
}

/// Local current-thread scheduler that bounds episodes and weighted resources.
pub struct LocalNativeGraphSuiteScheduler {
    limits: ResourceLimits,
    state: Rc<LocalSchedulerState>,
}

impl LocalNativeGraphSuiteScheduler {
    /// Creates one scheduler whose permits are released when each runner future completes.
    pub fn new(limits: ResourceLimits) -> Result<Self, MatrixError> {
        let (availability, _) = watch::channel(());
        Ok(Self {
            state: Rc::new(LocalSchedulerState {
                pools: RefCell::new(ResourcePools::from(&limits)),
                availability,
            }),
            limits,
        })
    }

    async fn run_local(
        &self,
        suite: ResolvedNativeGraphSuite,
        runner: Rc<dyn EpisodeRunner>,
    ) -> Result<Vec<EpisodeResult>, MatrixError> {
        let trials = suite.into_trials();
        let mut assignments = Vec::with_capacity(trials.len());
        for (output_index, trial) in trials.into_iter().enumerate() {
            assignments.push(EpisodeAssignment::new(output_index, trial));
        }
        validate_resource_requests(&self.limits, &assignments)?;

        // The stable slots exist before any concurrent runner future is admitted.
        let mut results = (0..assignments.len()).map(|_| None).collect::<Vec<_>>();
        let mut pending = VecDeque::from(assignments);
        let mut active = FuturesUnordered::new();
        let mut availability = self.state.availability.subscribe();

        while !pending.is_empty() || !active.is_empty() {
            let pending_count = pending.len();
            for _ in 0..pending_count {
                let Some(assignment) = pending.pop_front() else {
                    break;
                };
                let Some(lease) = self.try_admit(&assignment)? else {
                    pending.push_back(assignment);
                    continue;
                };
                let output_index = assignment.output_index();
                let expected_trial_digest = assignment.trial_digest().clone();
                let expected_attempt_id = assignment.attempt_id().clone();
                let runner = runner.clone();
                active.push(async move {
                    let result = runner.run(assignment).await;
                    drop(lease);
                    (
                        output_index,
                        expected_trial_digest,
                        expected_attempt_id,
                        result,
                    )
                });
            }

            if active.is_empty() {
                if pending.is_empty() {
                    break;
                }
                availability
                    .changed()
                    .await
                    .map_err(|_| MatrixError::AvailabilityNotificationsClosed)?;
                continue;
            }

            let completion = if pending.is_empty() {
                active.next().await
            } else {
                tokio::select! {
                    completion = active.next() => completion,
                    changed = availability.changed() => {
                        changed.map_err(|_| MatrixError::AvailabilityNotificationsClosed)?;
                        continue;
                    }
                }
            };
            let Some((output_index, expected_trial_digest, expected_attempt_id, result)) =
                completion
            else {
                return Err(MatrixError::MissingActiveEpisode);
            };
            let result = result?;
            if result.trial_digest() != &expected_trial_digest
                || result.attempt_id() != &expected_attempt_id
            {
                return Err(MatrixError::RunnerResultIdentityMismatch { output_index });
            }
            let assignment = results
                .get(output_index)
                .ok_or(MatrixError::MissingOutputSlot { output_index })?;
            if assignment.is_some() {
                return Err(MatrixError::DuplicateOutputSlot { output_index });
            }
            results[output_index] = Some(result);
        }

        let mut ordered = Vec::with_capacity(results.len());
        for (output_index, result) in results.into_iter().enumerate() {
            ordered.push(result.ok_or(MatrixError::MissingOutputSlot { output_index })?);
        }
        Ok(ordered)
    }

    fn try_admit(
        &self,
        assignment: &EpisodeAssignment,
    ) -> Result<Option<AdmissionLease>, MatrixError> {
        let mut pool = self.state.pools.borrow_mut();
        if !pool.try_acquire(assignment.resources()) {
            return Ok(None);
        }
        drop(pool);
        Ok(Some(AdmissionLease {
            state: self.state.clone(),
            resources: assignment.trial().resource_handle(),
        }))
    }
}

#[async_trait(?Send)]
impl NativeGraphSuiteScheduler for LocalNativeGraphSuiteScheduler {
    async fn run(
        &self,
        suite: ResolvedNativeGraphSuite,
        runner: Rc<dyn EpisodeRunner>,
    ) -> Result<Vec<EpisodeResult>, MatrixError> {
        self.run_local(suite, runner).await
    }
}

/// Runs a resolved suite through a selected narrow scheduler capability.
pub async fn run_resolved_suite(
    scheduler: &dyn NativeGraphSuiteScheduler,
    suite: ResolvedNativeGraphSuite,
    runner: Rc<dyn EpisodeRunner>,
) -> Result<Vec<EpisodeResult>, MatrixError> {
    scheduler.run(suite, runner).await
}

struct AdmissionLease {
    state: Rc<LocalSchedulerState>,
    resources: Rc<ResourceLeaseRequest>,
}

impl Drop for AdmissionLease {
    fn drop(&mut self) {
        self.state.pools.borrow_mut().release(&self.resources);
        self.state.availability.send_replace(());
    }
}

struct LocalSchedulerState {
    pools: RefCell<ResourcePools>,
    availability: watch::Sender<()>,
}

struct ResourcePools {
    episode_slots: usize,
    cpu_units: u32,
    memory_bytes: u64,
    model_binding_units: BTreeMap<ModelCapacityKey, u32>,
}

impl From<&ResourceLimits> for ResourcePools {
    fn from(limits: &ResourceLimits) -> Self {
        Self {
            episode_slots: limits.episode_slots,
            cpu_units: limits.cpu_units,
            memory_bytes: limits.memory_bytes,
            model_binding_units: limits.model_binding_units.clone(),
        }
    }
}

impl ResourcePools {
    fn try_acquire(&mut self, request: &ResourceLeaseRequest) -> bool {
        if self.episode_slots == 0
            || self.cpu_units < request.cpu_units()
            || self.memory_bytes < request.memory_bytes()
        {
            return false;
        }
        if request
            .model_binding_units()
            .iter()
            .any(|(binding, units)| {
                self.model_binding_units.get(binding).copied().unwrap_or(0) < *units
            })
        {
            return false;
        }
        self.episode_slots -= 1;
        self.cpu_units -= request.cpu_units();
        self.memory_bytes -= request.memory_bytes();
        for (binding, units) in request.model_binding_units() {
            if let Some(available) = self.model_binding_units.get_mut(binding) {
                *available -= *units;
            }
        }
        true
    }

    fn release(&mut self, request: &ResourceLeaseRequest) {
        self.episode_slots += 1;
        self.cpu_units += request.cpu_units();
        self.memory_bytes += request.memory_bytes();
        for (binding, units) in request.model_binding_units() {
            if let Some(available) = self.model_binding_units.get_mut(binding) {
                *available += *units;
            }
        }
    }
}

fn validate_resource_requests(
    limits: &ResourceLimits,
    assignments: &[EpisodeAssignment],
) -> Result<(), MatrixError> {
    for assignment in assignments {
        let request = assignment.resources();
        if request.cpu_units() > limits.cpu_units {
            return Err(MatrixError::ResourceRequestExceedsLimit {
                output_index: assignment.output_index(),
                resource: "cpu_units".to_owned(),
            });
        }
        if request.memory_bytes() > limits.memory_bytes {
            return Err(MatrixError::ResourceRequestExceedsLimit {
                output_index: assignment.output_index(),
                resource: "memory_bytes".to_owned(),
            });
        }
        for (binding, units) in request.model_binding_units() {
            let Some(capacity) = limits.model_binding_units.get(binding) else {
                return Err(MatrixError::MissingModelBindingCapacity {
                    output_index: assignment.output_index(),
                    binding: binding.digest().as_str().to_owned(),
                });
            };
            if units > capacity {
                return Err(MatrixError::ResourceRequestExceedsLimit {
                    output_index: assignment.output_index(),
                    resource: format!("model_binding_units:{}", binding.digest().as_str()),
                });
            }
        }
    }
    Ok(())
}

/// Failed local matrix admission or runner contract enforcement.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MatrixError {
    /// A global scheduler capacity was zero.
    ZeroCapacity(&'static str),
    /// An episode requested more of a resource than this scheduler owns.
    ResourceRequestExceedsLimit {
        /// Stable suite output index for the rejected attempt.
        output_index: usize,
        /// Human-readable resource key.
        resource: String,
    },
    /// An episode requested a package-scoped model capacity with no configured limit.
    MissingModelBindingCapacity {
        /// Stable suite output index for the rejected attempt.
        output_index: usize,
        /// Missing immutable capacity identity.
        binding: String,
    },
    /// An episode runner returned an execution failure before emitting a result.
    RunnerExecutionFailed(String),
    /// Scheduler-local admission notifications were unexpectedly closed.
    AvailabilityNotificationsClosed,
    /// An active episode set unexpectedly produced no completion.
    MissingActiveEpisode,
    /// A runner completion referenced no allocated stable output slot.
    MissingOutputSlot {
        /// Referenced output index.
        output_index: usize,
    },
    /// A runner completion tried to replace an earlier result.
    DuplicateOutputSlot {
        /// Output index that was already filled.
        output_index: usize,
    },
    /// A runner returned a result for another resolved trial or attempt.
    RunnerResultIdentityMismatch {
        /// Stable output index whose assignment the result did not satisfy.
        output_index: usize,
    },
}

impl fmt::Display for MatrixError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroCapacity(field) => {
                write!(
                    formatter,
                    "native graph matrix capacity {field} must be positive"
                )
            }
            Self::ResourceRequestExceedsLimit {
                output_index,
                resource,
            } => write!(
                formatter,
                "native graph matrix trial at output index {output_index} exceeds {resource} capacity"
            ),
            Self::MissingModelBindingCapacity {
                output_index,
                binding,
            } => write!(
                formatter,
                "native graph matrix trial at output index {output_index} requires unconfigured model binding {binding:?}"
            ),
            Self::RunnerExecutionFailed(message) => {
                write!(formatter, "native graph episode runner failed: {message}")
            }
            Self::AvailabilityNotificationsClosed => {
                formatter.write_str("native graph matrix admission notifications are closed")
            }
            Self::MissingActiveEpisode => {
                formatter.write_str("native graph matrix lost an active episode completion")
            }
            Self::MissingOutputSlot { output_index } => write!(
                formatter,
                "native graph matrix has no stable output slot {output_index}"
            ),
            Self::DuplicateOutputSlot { output_index } => write!(
                formatter,
                "native graph matrix would overwrite output slot {output_index}"
            ),
            Self::RunnerResultIdentityMismatch { output_index } => write!(
                formatter,
                "native graph runner result does not match assignment at output slot {output_index}"
            ),
        }
    }
}

impl std::error::Error for MatrixError {}
