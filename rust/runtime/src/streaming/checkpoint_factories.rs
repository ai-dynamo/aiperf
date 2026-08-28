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
        // `none` stores nothing, so it cannot protect anything. A closed-loop
        // run selects it only by disclaiming checkpoints entirely, which
        // `validate_target_policy` checks before the backend is consulted.
        if requirements.needs_sensitive_state_protection {
            return Err(configuration_error(
                "checkpoint backend \"none\" does not protect sensitive participant state at rest",
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

/// Registry identifier of the conditional object-store checkpoint backend.
#[cfg(feature = "streaming-s3")]
pub const OBJECT_STORE_CHECKPOINT_BACKEND_ID: &str =
    crate::streaming::checkpoints::object_store::OBJECT_STORE_CHECKPOINT_BACKEND_ID;

#[cfg(feature = "streaming-s3")]
static OBJECT_STORE_CHECKPOINT_BACKEND_DESCRIPTOR: StreamingCheckpointBackendDescriptor =
    StreamingCheckpointBackendDescriptor {
        id: OBJECT_STORE_CHECKPOINT_BACKEND_ID,
        description: "Conditional object-store generation pointer with bounded object I/O",
        is_durable: true,
        has_leased_readers: true,
        has_atomic_generations: true,
        has_result_segments: true,
        // Objects land with the bucket's own encryption policy; the backend
        // provides atomicity and reachability, not confidentiality at rest.
        protects_sensitive_state: false,
        retention: CheckpointRetention::GenerationReachability,
        // One conditional pointer is authoritative for every cell that can
        // reach the bucket, so the backend is shared rather than controller-local.
        placement: CheckpointBackendPlacement::SharedAcrossCells,
        // Provider round trips advance only with wall time.
        supports_virtual_clock: false,
    };

/// Strictly authored configuration for the object-store checkpoint backend.
#[cfg(feature = "streaming-s3")]
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObjectStoreCheckpointBackendConfig {
    /// Bucket owning every checkpoint object.
    pub bucket: String,
    /// Checkpoint prefix every object address derives from.
    pub prefix: String,
    /// Authored region; absent defers to the SDK region chain.
    #[serde(default)]
    pub region: Option<String>,
    /// Authored endpoint override for S3-compatible gateways.
    #[serde(default)]
    pub endpoint_url: Option<String>,
    /// Path-style addressing, required by most S3-compatible gateways.
    #[serde(default)]
    pub force_path_style: bool,
    /// Named credential profile; absent uses the default chain.
    #[serde(default)]
    pub profile: Option<String>,
    /// Maximum simultaneously retained objects per backend budget.
    #[serde(default = "default_max_items")]
    pub max_items: usize,
    /// Maximum simultaneously retained bytes per backend budget.
    #[serde(default = "default_max_bytes")]
    pub max_bytes: usize,
    /// Largest chunk one upload or restore retains from one provider response.
    #[serde(default = "default_max_chunk_bytes")]
    pub max_chunk_bytes: usize,
    /// Largest object sent in one `PutObject` request.
    #[serde(default = "default_single_put_threshold_bytes")]
    pub single_put_threshold_bytes: usize,
    /// Bytes buffered per multipart part above that threshold.
    #[serde(default = "default_multipart_part_bytes")]
    pub multipart_part_bytes: usize,
    /// Bounded per-operation-attempt timeout, in nanoseconds.
    #[serde(default = "default_operation_timeout_ns")]
    pub operation_timeout_ns: i64,
    /// Bounded connect timeout, in nanoseconds.
    #[serde(default = "default_connect_timeout_ns")]
    pub connect_timeout_ns: i64,
}

#[cfg(feature = "streaming-s3")]
const fn default_max_chunk_bytes() -> usize {
    8 * 1_024 * 1_024
}

#[cfg(feature = "streaming-s3")]
const fn default_single_put_threshold_bytes() -> usize {
    8 * 1_024 * 1_024
}

#[cfg(feature = "streaming-s3")]
const fn default_multipart_part_bytes() -> usize {
    8 * 1_024 * 1_024
}

#[cfg(feature = "streaming-s3")]
const fn default_operation_timeout_ns() -> i64 {
    30_000_000_000
}

#[cfg(feature = "streaming-s3")]
const fn default_connect_timeout_ns() -> i64 {
    10_000_000_000
}

/// Validated object-store configuration carrying its already-checked limits.
#[cfg(feature = "streaming-s3")]
#[derive(Debug)]
struct ValidatedObjectStoreConfig {
    client: crate::streaming::aws::AwsClientSettings,
    profile: Option<String>,
    store: crate::streaming::checkpoints::aws_object_store::AwsObjectStoreSettings,
    prefix: crate::streaming::checkpoints::object_store::ObjectKey,
    limits: crate::streaming::checkpoints::object_store::ObjectCheckpointLimits,
}

#[cfg(feature = "streaming-s3")]
impl ObjectStoreCheckpointBackendConfig {
    fn validate_config(&self) -> Result<ValidatedObjectStoreConfig, CheckpointError> {
        use crate::streaming::{
            aws::{AwsClientSettings, AwsProxySelection},
            checkpoints::{
                aws_object_store::AwsObjectStoreSettings,
                object_store::{ObjectCheckpointLimits, ObjectKey, ObjectListBudget},
            },
        };

        if self.bucket.is_empty() {
            return Err(configuration_error(
                "object-store checkpoint backend bucket must not be empty",
            ));
        }
        if self.prefix.is_empty() || self.prefix.ends_with('/') {
            return Err(configuration_error(
                "object-store checkpoint backend prefix must be nonempty and unterminated",
            ));
        }
        let nonzero = |value: usize, field: &str| {
            NonZeroUsize::new(value).ok_or_else(|| {
                configuration_error(format!(
                    "object-store checkpoint backend {field} must be greater than zero"
                ))
            })
        };
        let limits = BudgetLimits {
            max_items: nonzero(self.max_items, "max_items")?.get(),
            max_bytes: nonzero(self.max_bytes, "max_bytes")?.get(),
        };
        let prefix = ObjectKey::new(self.prefix.clone());
        Ok(ValidatedObjectStoreConfig {
            client: AwsClientSettings {
                region: self.region.clone(),
                endpoint_url: self.endpoint_url.clone(),
                force_path_style: self.force_path_style,
                // Checkpoint traffic never adopts the ambient proxy environment.
                proxy: AwsProxySelection::Disabled,
                operation_timeout_ns: self.operation_timeout_ns,
                connect_timeout_ns: self.connect_timeout_ns,
            },
            profile: self.profile.clone(),
            store: AwsObjectStoreSettings {
                bucket: self.bucket.clone(),
                prefix: prefix.clone(),
                max_retained_bytes: nonzero(self.max_bytes, "max_bytes")?,
                max_retained_items: nonzero(self.max_items, "max_items")?,
                single_put_threshold_bytes: nonzero(
                    self.single_put_threshold_bytes,
                    "single_put_threshold_bytes",
                )?,
                multipart_part_bytes: nonzero(self.multipart_part_bytes, "multipart_part_bytes")?,
            },
            prefix: prefix.clone(),
            limits: ObjectCheckpointLimits {
                transactions: limits,
                prepared_indexes: limits,
                storage: limits,
                result_summaries: limits,
                reads: limits,
                max_chunk_bytes: nonzero(self.max_chunk_bytes, "max_chunk_bytes")?,
                list: ObjectListBudget {
                    max_items: nonzero(self.max_items, "max_items")?,
                    max_metadata_bytes: nonzero(self.max_bytes, "max_bytes")?,
                },
            },
        })
    }
}

/// Startup validator and preparer for the object-store checkpoint backend.
#[cfg(feature = "streaming-s3")]
#[derive(Debug, Clone, Copy, Default)]
pub struct ObjectStoreCheckpointBackendFactory;

#[cfg(feature = "streaming-s3")]
impl StreamingCheckpointBackendFactory for ObjectStoreCheckpointBackendFactory {
    fn descriptor(&self) -> &'static StreamingCheckpointBackendDescriptor {
        &OBJECT_STORE_CHECKPOINT_BACKEND_DESCRIPTOR
    }

    fn validate(
        &self,
        authored: &RawValue,
        _requirements: &CheckpointBackendRequirements,
    ) -> Result<Box<dyn ValidatedCheckpointBackendConfig>, CheckpointError> {
        // A conditional pointer is durable and keeps partial results reachable
        // from committed roots, so it satisfies every requirement a run states.
        let config: ObjectStoreCheckpointBackendConfig = serde_json::from_str(authored.get())
            .map_err(|error| {
                configuration_error(format!("invalid object-store backend config: {error}"))
            })?;
        Ok(Box::new(config.validate_config()?))
    }

    fn prepare(
        &self,
        config: Box<dyn ValidatedCheckpointBackendConfig>,
        context: &CheckpointBackendPrepareContext,
    ) -> Result<Box<dyn StreamingCheckpointBackend>, CheckpointError> {
        use crate::streaming::checkpoints::{
            aws_object_store::LazyAwsConditionalObjectStore, object_store::ObjectCheckpointBackend,
        };

        let config = *config
            .into_any()
            .downcast::<ValidatedObjectStoreConfig>()
            .map_err(|_| configuration_error("object-store backend config type mismatch"))?;
        let store = LazyAwsConditionalObjectStore::new(
            config.client,
            config.profile,
            config.store,
            Rc::clone(&context.clock),
        )?;
        Ok(Box::new(ObjectCheckpointBackend::new(
            Rc::new(store),
            config.prefix,
            config.limits,
        )?))
    }
}
