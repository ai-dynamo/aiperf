// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Built-in checkpoint backend factories registered in the frozen registry.
//!
//! Two backends are compiled into the stock distribution: `local`, the
//! crash-durable on-disk store, and `none`, the explicit checkpoint-free
//! selection. Every other identifier fails closed at selection.

use std::{num::NonZeroUsize, path::PathBuf, rc::Rc};

use serde::Deserialize;
use serde_json::value::RawValue;

use crate::{
    clock::{Clock, RealClock},
    streaming::{
        blocking::StreamingBlockingExecutor,
        budget::BudgetLimits,
        checkpoint::{CheckpointError, CheckpointParticipantId},
        checkpoint_backend::{
            CheckpointBackendPlacement, CheckpointBackendPrepareContext,
            CheckpointBackendRequirements, CheckpointRetention, StreamingCheckpointBackend,
            StreamingCheckpointBackendDescriptor, StreamingCheckpointBackendFactory,
            ValidatedCheckpointBackendConfig,
        },
        checkpoints::{
            local::{BlockingLocalFilesystem, LocalCheckpointBackend, LocalCheckpointLimits},
            none::NoneCheckpointBackend,
        },
    },
};

/// Registry identifier of the crash-durable local checkpoint store.
pub const LOCAL_CHECKPOINT_BACKEND_ID: &str = "local";

/// Registry identifier of the checkpoint-free selection.
pub const NONE_CHECKPOINT_BACKEND_ID: &str = "none";

/// Stable participant identity of the local store's blocking filesystem owner.
const LOCAL_BLOCKING_PARTICIPANT: &str = "streaming-checkpoint-local-blocking";

/// Concurrent filesystem jobs the local store's blocking executor accepts.
const LOCAL_BLOCKING_JOBS: usize = 2;

static LOCAL_CHECKPOINT_BACKEND_DESCRIPTOR: StreamingCheckpointBackendDescriptor =
    StreamingCheckpointBackendDescriptor {
        id: LOCAL_CHECKPOINT_BACKEND_ID,
        description: "Crash-durable local generation store with leased readers",
        is_durable: true,
        has_leased_readers: true,
        has_atomic_generations: true,
        has_result_segments: true,
        // Objects land as ordinary files with the process umask; the store
        // provides atomicity and reachability, not confidentiality at rest.
        protects_sensitive_state: false,
        retention: CheckpointRetention::GenerationReachability,
        // Advisory locking is unreliable over shared filesystems, so the store
        // is authoritative for exactly one controller process.
        placement: CheckpointBackendPlacement::ControllerLocal,
        // The registered built-in binds a real clock at preparation because the
        // preparation context carries no clock; a virtual-clock run needs the
        // backend constructed directly with its own clock.
        supports_virtual_clock: false,
    };

static NONE_CHECKPOINT_BACKEND_DESCRIPTOR: StreamingCheckpointBackendDescriptor =
    StreamingCheckpointBackendDescriptor {
        id: NONE_CHECKPOINT_BACKEND_ID,
        description: "Checkpoint-free selection that publishes and retains nothing",
        is_durable: false,
        has_leased_readers: false,
        has_atomic_generations: false,
        has_result_segments: false,
        protects_sensitive_state: false,
        retention: CheckpointRetention::Ephemeral,
        placement: CheckpointBackendPlacement::ControllerLocal,
        supports_virtual_clock: true,
    };

fn configuration_error(message: impl Into<String>) -> CheckpointError {
    CheckpointError::Storage {
        message: message.into(),
    }
}

const fn default_max_items() -> usize {
    4_096
}

const fn default_max_bytes() -> usize {
    256 * 1_024 * 1_024
}

const fn default_gc_page_items() -> usize {
    64
}

const fn default_prepare_lease_ns() -> u64 {
    // Thirty seconds is long enough to outlive an ordinary staged transaction
    // and short enough that an abandoned scratch subtree becomes reclaimable
    // within one reasonable operator wait.
    30_000_000_000
}

/// Strictly authored configuration for the built-in local checkpoint store.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LocalCheckpointBackendConfig {
    /// Absolute directory owning every run's committed objects.
    pub root: PathBuf,
    /// Maximum simultaneously retained objects per backend budget.
    #[serde(default = "default_max_items")]
    pub max_items: usize,
    /// Maximum simultaneously retained bytes per backend budget.
    #[serde(default = "default_max_bytes")]
    pub max_bytes: usize,
    /// Maximum scratch entries examined per reclamation page.
    #[serde(default = "default_gc_page_items")]
    pub gc_page_items: usize,
    /// Lifetime granted to one prepare lease, in nanoseconds.
    #[serde(default = "default_prepare_lease_ns")]
    pub prepare_lease_ns: u64,
}

impl LocalCheckpointBackendConfig {
    /// Reject every authored value the store cannot honor, before any effect.
    fn validate(&self) -> Result<LocalCheckpointLimits, CheckpointError> {
        if !self.root.is_absolute() {
            return Err(configuration_error(
                "local checkpoint backend root must be an absolute path",
            ));
        }
        if self.max_items == 0 {
            return Err(configuration_error(
                "local checkpoint backend max_items must be greater than zero",
            ));
        }
        if self.max_bytes == 0 {
            return Err(configuration_error(
                "local checkpoint backend max_bytes must be greater than zero",
            ));
        }
        let gc_page_items = NonZeroUsize::new(self.gc_page_items).ok_or_else(|| {
            configuration_error("local checkpoint backend gc_page_items must be greater than zero")
        })?;
        if self.prepare_lease_ns == 0 {
            return Err(configuration_error(
                "local checkpoint backend prepare_lease_ns must be greater than zero",
            ));
        }
        let limits = BudgetLimits {
            max_items: self.max_items,
            max_bytes: self.max_bytes,
        };
        Ok(LocalCheckpointLimits {
            transactions: limits,
            prepared_indexes: limits,
            storage: limits,
            result_summaries: limits,
            reads: limits,
            gc_page_items,
            prepare_lease_ns: self.prepare_lease_ns,
        })
    }
}

/// Validated local configuration carrying its already-checked limits.
#[derive(Clone, Debug)]
struct ValidatedLocalConfig {
    root: PathBuf,
    limits: LocalCheckpointLimits,
}

/// Startup validator and preparer for the built-in local checkpoint store.
#[derive(Debug, Clone, Copy, Default)]
pub struct LocalCheckpointBackendFactory;

impl StreamingCheckpointBackendFactory for LocalCheckpointBackendFactory {
    fn descriptor(&self) -> &'static StreamingCheckpointBackendDescriptor {
        &LOCAL_CHECKPOINT_BACKEND_DESCRIPTOR
    }

    fn validate(
        &self,
        authored: &RawValue,
        _requirements: &CheckpointBackendRequirements,
    ) -> Result<Box<dyn ValidatedCheckpointBackendConfig>, CheckpointError> {
        // The local store is durable and keeps partial results reachable from
        // committed roots, so it satisfies every requirement a run can state.
        let config: LocalCheckpointBackendConfig =
            serde_json::from_str(authored.get()).map_err(|error| {
                configuration_error(format!("invalid local backend config: {error}"))
            })?;
        let limits = config.validate()?;
        Ok(Box::new(ValidatedLocalConfig {
            root: config.root,
            limits,
        }))
    }

    fn prepare(
        &self,
        config: Box<dyn ValidatedCheckpointBackendConfig>,
        context: &CheckpointBackendPrepareContext,
    ) -> Result<Box<dyn StreamingCheckpointBackend>, CheckpointError> {
        let config = *config
            .into_any()
            .downcast::<ValidatedLocalConfig>()
            .map_err(|_| configuration_error("local backend config type mismatch"))?;
        let executor = StreamingBlockingExecutor::new(
            context.run,
            CheckpointParticipantId::new(LOCAL_BLOCKING_PARTICIPANT),
            LOCAL_BLOCKING_JOBS,
            config.limits.storage.max_bytes,
            config.limits.storage.max_bytes,
        )
        .map_err(|error| {
            configuration_error(format!("local backend blocking executor: {error}"))
        })?;
        let backend = LocalCheckpointBackend::open(
            config.root,
            config.limits,
            Rc::new(BlockingLocalFilesystem::new(executor)),
            RealClock::new() as Rc<dyn Clock>,
        )?;
        Ok(Box::new(backend))
    }
}

/// Authored configuration for the checkpoint-free selection: no fields exist.
#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NoneCheckpointBackendConfig {}

/// Startup validator and preparer for the checkpoint-free selection.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoneCheckpointBackendFactory;

impl StreamingCheckpointBackendFactory for NoneCheckpointBackendFactory {
    fn descriptor(&self) -> &'static StreamingCheckpointBackendDescriptor {
        &NONE_CHECKPOINT_BACKEND_DESCRIPTOR
    }

    fn validate(
        &self,
        authored: &RawValue,
        requirements: &CheckpointBackendRequirements,
    ) -> Result<Box<dyn ValidatedCheckpointBackendConfig>, CheckpointError> {
        let config: NoneCheckpointBackendConfig =
            serde_json::from_str(authored.get()).map_err(|error| {
                configuration_error(format!("invalid none backend config: {error}"))
            })?;
        // A run that must survive process replacement cannot select a backend
        // that stores nothing; refusing here keeps the mismatch out of
        // execution rather than discovering it at the first missing head.
        if requirements.needs_restartable_execution {
            return Err(configuration_error(
                "checkpoint backend \"none\" cannot resume execution after process replacement",
            ));
        }
        if requirements.needs_durable_partial_results {
            return Err(configuration_error(
                "checkpoint backend \"none\" retains no durable partial results",
            ));
        }
        Ok(Box::new(config))
    }

    fn prepare(
        &self,
        config: Box<dyn ValidatedCheckpointBackendConfig>,
        _context: &CheckpointBackendPrepareContext,
    ) -> Result<Box<dyn StreamingCheckpointBackend>, CheckpointError> {
        config
            .into_any()
            .downcast::<NoneCheckpointBackendConfig>()
            .map_err(|_| configuration_error("none backend config type mismatch"))?;
        Ok(Box::new(NoneCheckpointBackend::new()))
    }
}
