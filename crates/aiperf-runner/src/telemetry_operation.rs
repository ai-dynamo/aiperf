// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Executable `online_http + telemetry_watch` runner pair.
//!
//! Static component/source validation is complete before this module resolves
//! key material, opens a target, qualifies a spool, or constructs control HTTP.
//! Collection and source-free finalization are separate prepared variants, so
//! the latter cannot accidentally activate source, parser, transport, or decode
//! machinery.

use std::collections::BTreeMap;
use std::fmt::{self, Debug, Formatter};
use std::io::Write;
use std::rc::Rc;
use std::sync::Arc;

use aiperf_clock::{Clock, RealClock};
use aiperf_metrics::{
    NativeReport, NativeReportInput, ReportPairRunFacts, ReportRunInfo, ReportTelemetryArchive,
    ReportTelemetryArchiveHead, ReportTelemetryArchiveHealth, ReportTelemetryArchiveSpoolBudget,
    ReportTelemetryArchiveState, ReportTelemetryBoundaryReference, ReportTelemetryBoundaryRole,
    ReportTelemetryLossKind, ReportTelemetryLossRange, ReportTelemetryLossReason,
    ReportTelemetryLossSaturationSummary, RunOutcome, TELEMETRY_ARCHIVE_REPORT_SCHEMA_VERSION,
};
use aiperf_prometheus::StrictExpositionParser;
use aiperf_telemetry_archive::{
    ArchiveFrameSequencerV1, ArchiveId, ArchiveProjectionFootprint, ArchiveRemoteSynchronizer,
    ArchiveSink, ArchiveSpoolBudgetAuthority, ArchiveSpoolBudgetLimits, ArchiveSpoolReservePlan,
    ArchiveSpoolResources, ArchiveState, AtomicArchiveSpoolBudget,
    CanonicalArchiveWalFrameDecoderV1, CanonicalJsonValue, ControlFrameCodecV1,
    DEFAULT_MAX_WAL_FRAME_BYTES, Digest, EpochAnchor, EpochAnchorProvider, ExecutionId,
    FixedLossLedgerV1, GenesisV1, LifecycleCompletionReasonV1, LocalArchiveRepository,
    LocalArchiveState, LossLedgerLimitsV1, NoDurabilityFaults, OwnedReceiptJournalMode,
    QualifiedSpool, ReceiptJournal, ReceiptObserverEpochV1, RecoveryPlan,
    RemotePublicationObservationV1, SessionAnchorV1, SessionId, SourceFrameCodecV1,
    SourceProjectionPolicyV1, SystemEpochAnchorProvider, TelemetryEnricherChain, TerminationReason,
    TimeDomain, WalSegmentHeaderV1, domain_digest,
};
use anyhow::{Context, Result, anyhow, bail, ensure};
use serde::Serialize;
use uuid::Uuid;

use crate::control_plane_http::{ControlPlaneClientPolicy, ControlPlaneHttpProviderFactory};
use crate::protocol_v2::{AuthoredRunSpecV2, RunDiagnosticArtifactV2};
use crate::redaction::redact_diagnostic;
use crate::registry::{
    OnlineHttpBackendConfigV2, PreparedRunFailure, PreparedRunOutcome, PreparedRunnerOperation,
    RunnerPairFactory, RunnerRegistryBuilder, RunnerRunContext, ValidatedBackendConfig,
    ValidatedWorkloadConfig,
};
use crate::telemetry_archive_components::{
    ArchiveCollectComponentPrepareContext, ArchiveComponentError, ArchiveKeyProviderResolver,
    ArchiveObjectStoreProvider, ArchiveRecoveryExpectation, ArchiveRecoveryOperation,
    EnvironmentArchiveKeyProviderResolver, NativeArchiveObjectStoreProvider,
    PreparedTelemetryArchiveCollectComponents, PreparedTelemetryArchiveSyncComponents,
};
use crate::telemetry_archive_owner::{
    ArchiveLifecycleObservation, TelemetryArchiveOwnerConfig, TelemetryArchiveOwnerFinalization,
    start_telemetry_archive_owner,
};
use crate::telemetry_execution::{ValidatedTelemetrySourceV2, ValidatedTelemetryWatchWorkloadV2};
use crate::telemetry_pipeline::{BoundedTelemetryDecodePool, PrometheusAttemptPipeline};
use crate::telemetry_source::ArchiveSourcePrepareContext;

const BACKEND_ID: &str = "online_http";
const WORKLOAD_ID: &str = "telemetry_watch";
const STANDALONE_LOSS_EXACT_RANGES: usize = 1_024;
const STANDALONE_LOSS_BOUNDARY_REFS_PER_RANGE: usize = 8;
const STANDALONE_LOSS_IDENTIFIER_BYTES: usize = 256;
const ARCHIVE_FAILURE_DIAGNOSTIC_PATH: &str = "archive-failure-diagnostic.json";
const MIB: u64 = 1 << 20;
const ARCHIVE_TABLE_COUNT: u64 = 6;
const SOURCE_TO_WAL_EXPANSION_BOUND: u64 = 24;
const INDEX_PATH_PAGE_BOUND: u64 = 16;

/// Register the fully executable standalone telemetry-watch pair.
pub fn register_online_http_telemetry_watch_pair(
    builder: &mut RunnerRegistryBuilder,
) -> Result<()> {
    register_online_http_telemetry_watch_pair_with_providers(
        builder,
        Arc::new(NativeArchiveObjectStoreProvider),
        Arc::new(EnvironmentArchiveKeyProviderResolver::default()),
    )
}

/// Register telemetry watch with deployment-selected store and key providers.
pub fn register_online_http_telemetry_watch_pair_with_providers(
    builder: &mut RunnerRegistryBuilder,
    store_provider: Arc<dyn ArchiveObjectStoreProvider>,
    key_resolver: Arc<dyn ArchiveKeyProviderResolver>,
) -> Result<()> {
    builder.register_pair(Arc::new(TelemetryWatchPairFactory {
        store_provider,
        key_resolver,
    }))
}

struct TelemetryWatchPairFactory {
    store_provider: Arc<dyn ArchiveObjectStoreProvider>,
    key_resolver: Arc<dyn ArchiveKeyProviderResolver>,
}

impl Debug for TelemetryWatchPairFactory {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TelemetryWatchPairFactory")
            .field("store_provider", &self.store_provider)
            .field("key_resolver", &self.key_resolver)
            .finish()
    }
}

impl RunnerPairFactory for TelemetryWatchPairFactory {
    fn backend_id(&self) -> &'static str {
        BACKEND_ID
    }

    fn workload_id(&self) -> &'static str {
        WORKLOAD_ID
    }

    fn validate_pair(
        &self,
        backend: &dyn ValidatedBackendConfig,
        workload: &dyn ValidatedWorkloadConfig,
    ) -> Result<()> {
        backend
            .as_any()
            .downcast_ref::<OnlineHttpBackendConfigV2>()
            .ok_or_else(|| anyhow!("telemetry watch received a non-HTTP backend config"))?;
        workload
            .as_any()
            .downcast_ref::<ValidatedTelemetryWatchWorkloadV2>()
            .ok_or_else(|| anyhow!("telemetry watch received another workload config"))?;
        Ok(())
    }

    fn validate_run(
        &self,
        run: &AuthoredRunSpecV2,
        _context: &RunnerRunContext,
        backend: &dyn ValidatedBackendConfig,
        workload: &dyn ValidatedWorkloadConfig,
    ) -> Result<()> {
        self.validate_pair(backend, workload)?;
        ensure!(
            !run.artifact_target.exists(),
            "telemetry watch artifact target already exists"
        );
        let workload = telemetry_workload(workload)?;
        let (spool, target) = match workload {
            ValidatedTelemetryWatchWorkloadV2::Collect { archive, .. } => {
                (&archive.local_spool, &archive.target)
            }
            ValidatedTelemetryWatchWorkloadV2::FinalizeRemote { archive, .. } => {
                (&archive.local_spool, &archive.target)
            }
        };
        ensure!(
            spool != &run.artifact_target,
            "telemetry archive spool must not alias the report artifact target"
        );
        if target.scheme() == "file" {
            let target_path = url::Url::parse(target.as_str())
                .map_err(|error| anyhow!("validated archive target is invalid: {error}"))?
                .to_file_path()
                .map_err(|()| anyhow!("validated file archive target has no local path"))?;
            ensure!(
                &target_path != spool && target_path != run.artifact_target,
                "telemetry archive target, spool, and report artifacts must be distinct"
            );
        }
        Ok(())
    }

    fn prepare(
        &self,
        _run: &AuthoredRunSpecV2,
        _backend: Box<dyn ValidatedBackendConfig>,
        _workload: Box<dyn ValidatedWorkloadConfig>,
    ) -> Result<Box<dyn PreparedRunnerOperation>> {
        bail!("telemetry watch requires the coordinator-owned runner context")
    }

    fn prepare_with_context(
        &self,
        run: &AuthoredRunSpecV2,
        context: &RunnerRunContext,
        backend: Box<dyn ValidatedBackendConfig>,
        workload: Box<dyn ValidatedWorkloadConfig>,
    ) -> Result<Box<dyn PreparedRunnerOperation>> {
        self.validate_pair(backend.as_ref(), workload.as_ref())?;
        let distribution_id = parse_distribution_digest(context.distribution_id())?;
        let backend = backend
            .into_any()
            .downcast::<OnlineHttpBackendConfigV2>()
            .map_err(|_| anyhow!("telemetry watch lost its validated HTTP backend type"))?;
        let workload = workload
            .into_any()
            .downcast::<ValidatedTelemetryWatchWorkloadV2>()
            .map_err(|_| anyhow!("telemetry watch lost its validated workload type"))?;
        let run_id = run.identity.benchmark_id.clone();
        let prepared = match *workload {
            ValidatedTelemetryWatchWorkloadV2::Collect {
                duration_ns,
                shutdown_timeout_ns,
                sources,
                archive,
            } => {
                let archive = archive
                    .prepare(ArchiveCollectComponentPrepareContext {
                        store_provider: self.store_provider.as_ref(),
                        key_resolver: self.key_resolver.as_ref(),
                        durability_faults: Arc::new(NoDurabilityFaults),
                    })
                    .map_err(component_error)?;
                PreparedTelemetryWatchOperation::Collect(PreparedCollect {
                    run_id,
                    artifact_target: run.artifact_target.clone(),
                    distribution_id,
                    duration_ns,
                    shutdown_timeout_ns,
                    sources,
                    archive,
                    control_plane_policy: ControlPlaneClientPolicy {
                        connect_timeout_ns: backend.client.connect_timeout_ns,
                    },
                    control_plane_factory: context
                        .execution_factories()
                        .control_plane_http_handle(),
                })
            }
            ValidatedTelemetryWatchWorkloadV2::FinalizeRemote {
                shutdown_timeout_ns,
                archive,
            } => {
                let archive = archive
                    .prepare(
                        crate::telemetry_archive_components::ArchiveSyncComponentPrepareContext {
                            store_provider: self.store_provider.as_ref(),
                            key_resolver: self.key_resolver.as_ref(),
                        },
                    )
                    .map_err(component_error)?;
                PreparedTelemetryWatchOperation::FinalizeRemote(PreparedSync {
                    run_id,
                    artifact_target: run.artifact_target.clone(),
                    distribution_id,
                    shutdown_timeout_ns,
                    archive,
                })
            }
        };
        Ok(Box::new(prepared))
    }
}

struct PreparedCollect {
    run_id: String,
    artifact_target: std::path::PathBuf,
    distribution_id: Digest,
    duration_ns: Option<i64>,
    shutdown_timeout_ns: i64,
    sources: Vec<ValidatedTelemetrySourceV2>,
    archive: PreparedTelemetryArchiveCollectComponents,
    control_plane_policy: ControlPlaneClientPolicy,
    control_plane_factory: Arc<dyn ControlPlaneHttpProviderFactory>,
}

struct PreparedSync {
    run_id: String,
    artifact_target: std::path::PathBuf,
    distribution_id: Digest,
    shutdown_timeout_ns: i64,
    archive: PreparedTelemetryArchiveSyncComponents,
}

enum PreparedTelemetryWatchOperation {
    Collect(PreparedCollect),
    FinalizeRemote(PreparedSync),
}

impl Debug for PreparedTelemetryWatchOperation {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Collect(prepared) => formatter
                .debug_struct("PreparedTelemetryWatchOperation::Collect")
                .field("run_id", &prepared.run_id)
                .field("duration_ns", &prepared.duration_ns)
                .field("shutdown_timeout_ns", &prepared.shutdown_timeout_ns)
                .field("source_count", &prepared.sources.len())
                .field("archive", &prepared.archive)
                .finish(),
            Self::FinalizeRemote(prepared) => formatter
                .debug_struct("PreparedTelemetryWatchOperation::FinalizeRemote")
                .field("run_id", &prepared.run_id)
                .field("shutdown_timeout_ns", &prepared.shutdown_timeout_ns)
                .field("archive", &prepared.archive)
                .finish(),
        }
    }
}

impl PreparedRunnerOperation for PreparedTelemetryWatchOperation {
    fn execute(self: Box<Self>) -> Result<PreparedRunOutcome> {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .context("building telemetry-watch runtime")?;
        let local = tokio::task::LocalSet::new();
        let outcome = match *self {
            Self::Collect(prepared) => {
                runtime.block_on(local.run_until(execute_collect(prepared)))?
            }
            Self::FinalizeRemote(prepared) => {
                runtime.block_on(local.run_until(execute_sync(prepared)))?
            }
        };
        Ok(outcome)
    }
}

async fn execute_collect(mut prepared: PreparedCollect) -> Result<PreparedRunOutcome> {
    prepare_artifact_target(&prepared.artifact_target)?;
    let clock: Rc<dyn Clock> = RealClock::new();
    let epoch_anchor = SystemEpochAnchorProvider::default()
        .anchor(clock.as_ref())
        .context("capturing telemetry epoch anchor")?;
    let session_anchor = SessionAnchorV1::new(TimeDomain::Real, Some(epoch_anchor))?;
    let session_id = session_id(Uuid::new_v4())?;
    let execution_uuid = Uuid::new_v4();
    let execution_id = ExecutionId::new(*execution_uuid.as_bytes())?;
    let source_descriptors = source_descriptors(&prepared.sources)?;
    let archive_identity_digest = collect_identity_digest(&prepared.archive, &source_descriptors)?;
    let archive_target_digest = prepared.archive.archive_target_digest();
    let archive_key_digest = prepared
        .archive
        .persistent_component_identities
        .iter()
        .find(|identity| identity.family == "archive_key")
        .ok_or_else(|| anyhow!("prepared archive omitted archive-key identity"))?
        .digest;
    let recovery_operation = prepared.archive.recovery.operation();
    let archive_id = match recovery_operation {
        ArchiveRecoveryOperation::CreateNew => archive_id(Uuid::new_v4())?,
        ArchiveRecoveryOperation::ExactResume => prepared
            .archive
            .recovery
            .expected_archive_id()
            .ok_or_else(|| anyhow!("exact resume omitted its authored archive ID"))?,
        ArchiveRecoveryOperation::FinalizeRemote => {
            bail!("source-free recovery cannot activate telemetry collection")
        }
    };
    let recovery = prepared
        .archive
        .recovery
        .bind_collect(ArchiveRecoveryExpectation {
            archive_id,
            persistent_identity_digest: archive_identity_digest,
            archive_target_digest,
        })
        .map_err(component_error)?;
    let prior_writer_claim_id = recovery.prior_writer_claim_id();
    let spool = QualifiedSpool::open(&prepared.archive.local_spool)
        .context("qualifying telemetry archive spool")?;
    let synchronizer = ArchiveRemoteSynchronizer::new(prepared.archive.store_access.store.clone())
        .context("validating archive publication capabilities")?;
    let receipt_epoch = receipt_epoch(
        execution_id,
        Some(session_id),
        epoch_anchor,
        prepared.distribution_id,
    )?;
    let schemas = prepared.archive.writer.schemas().clone();
    let (sink, first_record_seq, mut next_receipt_seq): (Box<dyn ArchiveSink>, u64, u64) =
        match recovery_operation {
            ArchiveRecoveryOperation::CreateNew => {
                ensure!(
                    recovery.recover(&LocalArchiveState::Absent, None)? == RecoveryPlan::CreateNew,
                    "create-new recovery policy returned another operation"
                );
                let canonical_spool_id = domain_digest(
                    "aiperf.archive.canonical-spool.v1",
                    &[
                        Uuid::new_v4().as_bytes(),
                        prepared.archive.local_spool.as_os_str().as_encoded_bytes(),
                    ],
                );
                let genesis = GenesisV1 {
                    archive_id,
                    canonical_spool_id,
                    archive_identity_digest,
                    archive_target_digest,
                    archive_key_digest,
                    writer_compatibility_id: prepared.archive.writer.writer_compatibility_id(),
                    runner_distribution_id: prepared.distribution_id,
                    source_descriptors: source_descriptors.clone(),
                    persistent_writer_identity: prepared
                        .archive
                        .writer
                        .persistent_writer_identity()
                        .clone(),
                    initial_session_id: Some(session_id),
                    time_domain: TimeDomain::Real,
                    epoch_anchor: Some(epoch_anchor),
                };
                let repository =
                    LocalArchiveRepository::create_new(spool, genesis, &NoDurabilityFaults)
                        .context("committing telemetry archive genesis")?;
                let first_record_seq = repository.head().next_record_seq;
                let header = WalSegmentHeaderV1::new(
                    archive_id,
                    session_id,
                    repository.head().generation_hash,
                    repository.head().genesis_hash,
                    prepared.archive.writer.writer_compatibility_id(),
                    first_record_seq,
                    session_anchor,
                    schemas
                        .iter()
                        .map(|schema| (schema.table(), schema.fingerprint()))
                        .collect(),
                )?;
                let sink = prepared
                    .archive
                    .writer
                    .prepare_new_sink(
                        repository,
                        header,
                        OwnedReceiptJournalMode::Bootstrap(receipt_epoch.clone()),
                    )
                    .map_err(component_error)?;
                (sink, first_record_seq, 0)
            }
            ArchiveRecoveryOperation::ExactResume => {
                let repository = LocalArchiveRepository::recover_existing(
                    spool,
                    archive_id,
                    archive_target_digest,
                    &NoDurabilityFaults,
                )
                .context("recovering exact telemetry archive authority")?;
                ensure!(
                    repository.genesis().archive_identity_digest == archive_identity_digest,
                    "exact resume persistent archive identity differs from genesis"
                );
                ensure!(
                    repository.genesis().archive_key_digest == archive_key_digest,
                    "exact resume archive-key identity differs from genesis"
                );
                ensure!(
                    repository.genesis().writer_compatibility_id
                        == prepared.archive.writer.writer_compatibility_id(),
                    "exact resume writer compatibility differs from genesis"
                );
                ensure!(
                    repository.genesis().source_descriptors == source_descriptors,
                    "exact resume source descriptors differ from genesis"
                );
                ensure!(
                    repository.head().archive_state == ArchiveState::Open,
                    "exact collect resume requires an open local archive"
                );
                let local_state = LocalArchiveState::Verified {
                    archive_id,
                    persistent_identity_digest: archive_identity_digest,
                    head_hash: repository.head().generation_hash,
                    archive_state: repository.head().archive_state,
                    session_id: repository.latest_collection_session_id(),
                    next_record_seq: repository.head().next_record_seq,
                };
                ensure!(
                    matches!(
                        recovery.recover(&local_state, None)?,
                        RecoveryPlan::ResumeLocal { .. } | RecoveryPlan::ResumeAndPublish { .. }
                    ),
                    "exact-resume recovery policy returned another operation"
                );
                let receipt_pointer = repository.spool().path().join("LOCAL-RECEIPTS");
                let (receipt_mode, next_receipt_seq) = if receipt_pointer
                    .try_exists()
                    .context("checking telemetry receipt authority")?
                {
                    let receipts = ReceiptJournal::recover(
                        repository.spool(),
                        archive_id,
                        &NoDurabilityFaults,
                    )
                    .context("recovering telemetry publication receipts")?;
                    let next_receipt_seq =
                        receipts.last_receipt_seq().map_or(Ok(0), |sequence| {
                            sequence
                                .checked_add(1)
                                .ok_or_else(|| anyhow!("telemetry receipt sequence overflowed"))
                        })?;
                    drop(receipts);
                    (
                        OwnedReceiptJournalMode::Recover {
                            observer_epoch: Some(receipt_epoch.clone()),
                        },
                        next_receipt_seq,
                    )
                } else {
                    (OwnedReceiptJournalMode::Bootstrap(receipt_epoch.clone()), 0)
                };
                let decoder = CanonicalArchiveWalFrameDecoderV1::with_schemas(schemas.clone());
                let sink = prepared
                    .archive
                    .writer
                    .prepare_resumed_sink(
                        repository,
                        session_id,
                        session_anchor,
                        DEFAULT_MAX_WAL_FRAME_BYTES,
                        receipt_mode,
                        &decoder,
                    )
                    .map_err(component_error)?;
                let first_record_seq = sink
                    .local_repository()
                    .ok_or_else(|| anyhow!("resumed archive sink omitted local authority"))?
                    .head()
                    .next_record_seq;
                (sink, first_record_seq, next_receipt_seq)
            }
            ArchiveRecoveryOperation::FinalizeRemote => unreachable!("checked above"),
        };
    let publication_observation = RemotePublicationObservationV1 {
        observer_epoch_id: receipt_epoch.observer_epoch_id,
    };
    let repository = sink
        .local_repository()
        .ok_or_else(|| anyhow!("archive sink omitted its live local authority"))?;
    let active_publication = match recovery_operation {
        ArchiveRecoveryOperation::CreateNew => {
            synchronizer
                .publish_active(
                    repository,
                    session_id,
                    publication_observation,
                    clock.as_ref(),
                )
                .await
        }
        ArchiveRecoveryOperation::ExactResume => {
            synchronizer
                .publish_resumed_active(
                    repository,
                    session_id,
                    prior_writer_claim_id,
                    publication_observation,
                    clock.as_ref(),
                )
                .await
        }
        ArchiveRecoveryOperation::FinalizeRemote => unreachable!("checked above"),
    };
    match active_publication {
        Ok(active_publication) => {
            let active_receipt = active_publication
                .receipt
                .ok_or_else(|| anyhow!("active archive publication omitted its durable receipt"))?;
            ensure!(
                active_receipt.receipt_seq == next_receipt_seq,
                "active publication receipt sequence disagrees with recovered journal authority"
            );
            next_receipt_seq = next_receipt_seq
                .checked_add(1)
                .ok_or_else(|| anyhow!("telemetry receipt sequence overflowed"))?;
        }
        Err(error)
            if recovery_operation == ArchiveRecoveryOperation::CreateNew
                && !prepared.archive.required =>
        {
            // A new optional archive has no remote ancestry to fence. Its
            // qualified spool remains the sole authority until a later bounded
            // retry or source-free finalization publishes the active claim.
            let _ = error;
        }
        Err(error) => {
            return Err(archive_reporting_failure(
                &prepared.artifact_target,
                "archive_activation_failed",
                format!("publishing active archive claim failed: {error}"),
                Some(archive_id),
                Some(repository.head()),
            ));
        }
    }
    let data_queue_capacity = prepared.sources.len().saturating_mul(2).max(4);
    let control_queue_capacity = prepared.sources.len().max(4);
    let (budget_authority, ordinary_projection_footprint, control_projection_footprint) =
        prepare_spool_budget(
            prepared.archive.rotation.parquet.target_uncompressed_bytes,
            prepared
                .archive
                .raw_body
                .envelope()
                .map(|envelope| envelope.limits().max_plaintext_bytes()),
            prepared.archive.spool_quota_bytes,
            prepared.archive.spool_quota_files,
            &prepared.sources,
            repository.spool(),
            data_queue_capacity,
            control_queue_capacity,
        )?;
    let source_policies = prepared
        .sources
        .iter()
        .map(|source| {
            (
                source.id.clone(),
                SourceProjectionPolicyV1 {
                    attributes: source.attributes.clone(),
                },
            )
        })
        .collect();
    let global_enricher: Arc<dyn aiperf_telemetry_archive::TelemetryEnricher> = Arc::new(
        TelemetryEnricherChain::new(std::mem::take(&mut prepared.archive.enrichers)),
    );
    let sequencer = ArchiveFrameSequencerV1::with_projection_policies_at_record_seq(
        archive_id,
        session_id,
        Some(epoch_anchor),
        prepared.archive.archive_key.clone(),
        source_policies,
        global_enricher,
        prepared.archive.sanitizer.clone(),
        first_record_seq,
    )?;
    let loss_ledger = FixedLossLedgerV1::new(
        archive_id,
        session_id,
        prepared.sources.iter().map(|source| source.id.clone()),
        LossLedgerLimitsV1 {
            max_exact_ranges: STANDALONE_LOSS_EXACT_RANGES,
            max_sources: prepared.sources.len(),
            max_source_id_bytes: STANDALONE_LOSS_IDENTIFIER_BYTES,
            max_boundary_refs_per_range: STANDALONE_LOSS_BOUNDARY_REFS_PER_RANGE,
            max_boundary_identifier_bytes: STANDALONE_LOSS_IDENTIFIER_BYTES,
        },
    )?;
    let (source_owner, running_owner) =
        start_telemetry_archive_owner(TelemetryArchiveOwnerConfig {
            archive_id,
            session_id,
            sequencer,
            codec: SourceFrameCodecV1::with_schemas(schemas.clone()),
            control_codec: ControlFrameCodecV1::with_schemas(schemas),
            sink,
            clock: clock.clone(),
            receipt_epoch: receipt_epoch.clone(),
            receipt_epoch_registered: true,
            next_receipt_seq,
            loss_ledger,
            queue_capacity: data_queue_capacity,
            control_queue_capacity,
            best_effort: false,
            attached: false,
            budget_authority,
            admission_policy: prepared.archive.admission.clone(),
            ordinary_projection_footprint,
            control_projection_footprint,
        })
        .await
        .context("starting telemetry archive owner")?;
    running_owner
        .observe_lifecycle(ArchiveLifecycleObservation::session_started(clock.now_ns()))
        .await
        .context("persisting telemetry session-start marker")?;

    let run_start_ns = clock.now_ns();
    let run_deadline_ns = prepared
        .duration_ns
        .map(|duration| checked_deadline(run_start_ns, duration))
        .transpose()?;
    let control_plane = prepared
        .control_plane_factory
        .prepare(clock.clone(), prepared.control_plane_policy);
    let decode_pool = BoundedTelemetryDecodePool::new(prepared.sources.len().clamp(1, 8))?;
    let parser: Arc<dyn aiperf_prometheus::ExpositionParser> = Arc::new(StrictExpositionParser);
    let mut drivers = Vec::with_capacity(prepared.sources.len());
    for source in prepared.sources {
        let pipeline: Rc<dyn aiperf_telemetry_archive::TelemetryAttemptConsumer> =
            Rc::new(PrometheusAttemptPipeline::strict_standalone(
                clock.clone(),
                parser.clone(),
                aiperf_telemetry_archive::DecodeLimits::default(),
                decode_pool.clone(),
                source_owner.clone(),
            )?);
        let driver = source.prepared.prepare(ArchiveSourcePrepareContext {
            source_id: source.id,
            interval_ns: source.interval_ns,
            request_timeout_ns: source.request_timeout_ns,
            run_deadline_ns,
            clock: clock.clone(),
            control_plane: control_plane.clone(),
            consumer: pipeline,
        })?;
        drivers.push(driver.start()?);
    }

    let termination = wait_for_stop(clock.clone(), run_start_ns, prepared.duration_ns).await?;
    let shutdown_deadline = checked_deadline(clock.now_ns(), prepared.shutdown_timeout_ns)?;
    for driver in &drivers {
        driver.stop(shutdown_deadline);
    }
    let mut driver_attempts = 0_u64;
    for driver in drivers {
        let summary = await_before(clock.clone(), shutdown_deadline, driver.join())
            .await
            .context("draining telemetry source")??;
        driver_attempts = driver_attempts
            .checked_add(summary.attempts)
            .ok_or_else(|| anyhow!("telemetry driver attempt count overflowed"))?;
    }
    let completion_reason = match termination {
        TerminationReason::Duration => LifecycleCompletionReasonV1::Duration,
        TerminationReason::Signal | TerminationReason::Requested => {
            LifecycleCompletionReasonV1::Shutdown
        }
        TerminationReason::Failure | TerminationReason::SyncOnly => {
            LifecycleCompletionReasonV1::Failed
        }
    };
    running_owner
        .observe_lifecycle(ArchiveLifecycleObservation::session_stopped(
            clock.now_ns(),
            completion_reason,
        ))
        .await
        .context("persisting telemetry session-stop marker")?;
    let local = await_before(
        clock.clone(),
        shutdown_deadline,
        running_owner.finalize(termination),
    )
    .await;
    let mut local = match local {
        Ok(Ok(local)) => local,
        Ok(Err(error)) => {
            return Err(archive_reporting_failure(
                &prepared.artifact_target,
                "archive_local_finalization_failed",
                format!("telemetry local finalization failed: {error}"),
                Some(archive_id),
                None,
            ));
        }
        Err(error) => {
            return Err(archive_reporting_failure(
                &prepared.artifact_target,
                "archive_local_finalization_deadline",
                format!("telemetry local finalization deadline failed: {error:#}"),
                Some(archive_id),
                None,
            ));
        }
    };
    if local.completion.is_none() || !local.summary.writer_alive {
        return Err(archive_reporting_failure(
            &prepared.artifact_target,
            "archive_local_finalization_failed",
            local
                .summary
                .first_failure
                .clone()
                .unwrap_or_else(|| "telemetry local finalization produced no durable head".to_owned()),
            Some(archive_id),
            local.repository.as_ref().map(LocalArchiveRepository::head),
        ));
    }
    let repository = local
        .repository
        .take()
        .ok_or_else(|| anyhow!("local archive sink did not transfer its repository"))?;
    let remote = await_before(
        clock.clone(),
        shutdown_deadline,
        synchronizer.finalize_remote(
            &repository,
            session_id,
            RemotePublicationObservationV1 {
                observer_epoch_id: receipt_epoch.observer_epoch_id,
            },
            clock.as_ref(),
        ),
    )
    .await;
    let remote_head = match remote {
        Ok(Ok(remote)) => Some(remote.remote.head),
        Ok(Err(error)) if !prepared.archive.required => {
            let _ = error;
            None
        }
        Err(error) if !prepared.archive.required => {
            let _ = error;
            None
        }
        Ok(Err(error)) => {
            return Err(archive_reporting_failure(
                &prepared.artifact_target,
                "archive_remote_finalization_failed",
                format!("telemetry remote finalization failed: {error}"),
                Some(archive_id),
                Some(repository.head()),
            ));
        }
        Err(error) => {
            return Err(archive_reporting_failure(
                &prepared.artifact_target,
                "archive_remote_finalization_deadline",
                format!("telemetry remote finalization deadline failed: {error:#}"),
                Some(archive_id),
                Some(repository.head()),
            ));
        }
    };
    let report = archive_report(
        archive_id,
        execution_uuid,
        receipt_epoch.observer_epoch_id.digest(),
        Some(session_id),
        Some(session_id),
        &prepared.archive.local_spool,
        &prepared.archive.target,
        local,
        remote_head,
    )?;
    prepared_outcome(prepared.run_id, report, driver_attempts)
}

async fn execute_sync(prepared: PreparedSync) -> Result<PreparedRunOutcome> {
    prepare_artifact_target(&prepared.artifact_target)?;
    let archive_target_digest = prepared.archive.archive_target_digest();
    prepared
        .archive
        .recovery
        .bind_finalize_remote()
        .map_err(component_error)?;
    let clock: Rc<dyn Clock> = RealClock::new();
    let epoch_anchor = SystemEpochAnchorProvider::default().anchor(clock.as_ref())?;
    let execution_uuid = Uuid::new_v4();
    let execution_id = ExecutionId::new(*execution_uuid.as_bytes())?;
    let archive_id = archive_id(prepared.archive.archive_id)?;
    let spool = QualifiedSpool::open(&prepared.archive.local_spool)?;
    let repository = LocalArchiveRepository::recover_existing(
        spool,
        archive_id,
        archive_target_digest,
        &NoDurabilityFaults,
    )?;
    let key_identity = prepared.archive.invocation_component_identities[2].digest;
    ensure!(
        repository.genesis().archive_key_digest == key_identity,
        "source-free archive key selector does not match durable genesis"
    );
    let receipt_epoch = receipt_epoch(execution_id, None, epoch_anchor, prepared.distribution_id)?;
    let mut receipts =
        ReceiptJournal::recover(repository.spool(), archive_id, &NoDurabilityFaults)?;
    receipts.append_observer_epoch(receipt_epoch.clone(), &NoDurabilityFaults)?;
    drop(receipts);
    let latest_session_id = repository
        .latest_collection_session_id()
        .ok_or_else(|| anyhow!("verified archive has no collection session"))?;
    let synchronizer = ArchiveRemoteSynchronizer::new(prepared.archive.store_access.store.clone())?;
    let deadline = checked_deadline(clock.now_ns(), prepared.shutdown_timeout_ns)?;
    let remote = await_before(
        clock.clone(),
        deadline,
        synchronizer.finalize_remote(
            &repository,
            latest_session_id,
            RemotePublicationObservationV1 {
                observer_epoch_id: receipt_epoch.observer_epoch_id,
            },
            clock.as_ref(),
        ),
    )
    .await;
    let remote = match remote {
        Ok(Ok(remote)) => remote,
        Ok(Err(error)) => {
            return Err(archive_reporting_failure(
                &prepared.artifact_target,
                "archive_remote_finalization_failed",
                format!("source-free remote finalization failed: {error}"),
                Some(archive_id),
                Some(repository.head()),
            ));
        }
        Err(error) => {
            return Err(archive_reporting_failure(
                &prepared.artifact_target,
                "archive_remote_finalization_deadline",
                format!("source-free remote finalization deadline failed: {error:#}"),
                Some(archive_id),
                Some(repository.head()),
            ));
        }
    };
    let local_head = repository.head().clone();
    let report = ReportTelemetryArchive {
        schema_version: TELEMETRY_ARCHIVE_REPORT_SCHEMA_VERSION.to_owned(),
        archive_id: prepared.archive.archive_id.to_string(),
        execution_id: execution_uuid.to_string(),
        receipt_observer_epoch_id: receipt_epoch.observer_epoch_id.digest().to_tagged_hex(),
        collection_session_id: None,
        latest_collection_session_id: Some(uuid_session(latest_session_id)),
        state: ReportTelemetryArchiveState::RemotelyFinalized,
        publication_receipts_uri: local_file_uri(
            &prepared.archive.local_spool.join("LOCAL-RECEIPTS"),
        )?,
        local_head: Some(local_report_head(
            &prepared.archive.local_spool,
            &local_head,
        )?),
        remote_head: Some(remote_report_head(
            &prepared.archive.target,
            &remote.remote.head,
        )),
        finalized_local: true,
        finalized_remote: true,
        lossy: false,
        health: ReportTelemetryArchiveHealth {
            loss_ranges: Vec::new(),
            loss_saturation_summaries: Vec::new(),
            complete_ranges: true,
            writer_alive: true,
            spool_budget: None,
        },
    };
    prepared_outcome(prepared.run_id, report, 0)
}

fn archive_report(
    archive_id: ArchiveId,
    execution_uuid: Uuid,
    receipt_epoch_id: Digest,
    collection_session_id: Option<SessionId>,
    latest_collection_session_id: Option<SessionId>,
    spool: &std::path::Path,
    target: &crate::telemetry_watch::NormalizedArchiveUri,
    local: TelemetryArchiveOwnerFinalization,
    remote_head: Option<aiperf_telemetry_archive::HeadDescriptorV1>,
) -> Result<ReportTelemetryArchive> {
    let local_head = local
        .completion
        .as_ref()
        .ok_or_else(|| anyhow!("archive local finalization did not complete"))?
        .local_head
        .as_ref()
        .ok_or_else(|| anyhow!("archive local finalization omitted its head"))?;
    let health = owner_report_health(
        local
            .summary
            .loss_ledger
            .as_ref()
            .ok_or_else(|| anyhow!("archive owner omitted its bounded loss-ledger view"))?,
        local.summary.writer_alive,
        local.summary.budget.as_ref(),
    );
    let lossy = !health.writer_alive
        || !health.loss_ranges.is_empty()
        || !health.loss_saturation_summaries.is_empty();
    Ok(ReportTelemetryArchive {
        schema_version: TELEMETRY_ARCHIVE_REPORT_SCHEMA_VERSION.to_owned(),
        archive_id: uuid_archive(archive_id),
        execution_id: execution_uuid.to_string(),
        receipt_observer_epoch_id: receipt_epoch_id.to_tagged_hex(),
        collection_session_id: collection_session_id.map(uuid_session),
        latest_collection_session_id: latest_collection_session_id.map(uuid_session),
        state: if remote_head.is_some() {
            ReportTelemetryArchiveState::RemotelyFinalized
        } else {
            ReportTelemetryArchiveState::LocallyFinalized
        },
        publication_receipts_uri: local_file_uri(&spool.join("LOCAL-RECEIPTS"))?,
        local_head: Some(local_report_head(spool, local_head)?),
        remote_head: remote_head
            .as_ref()
            .map(|head| remote_report_head(target, head)),
        finalized_local: true,
        finalized_remote: remote_head.is_some(),
        lossy,
        health,
    })
}

fn owner_report_health(
    view: &aiperf_telemetry_archive::LossLedgerViewV1,
    writer_alive: bool,
    budget: Option<&aiperf_telemetry_archive::ArchiveSpoolBudgetSnapshot>,
) -> ReportTelemetryArchiveHealth {
    ReportTelemetryArchiveHealth {
        loss_ranges: view.exact_ranges.iter().map(report_loss_range).collect(),
        loss_saturation_summaries: view
            .saturation_snapshots
            .iter()
            .map(report_loss_saturation)
            .collect(),
        complete_ranges: view.complete_ranges,
        writer_alive,
        spool_budget: budget.map(report_spool_budget),
    }
}

fn report_spool_budget(
    budget: &aiperf_telemetry_archive::ArchiveSpoolBudgetSnapshot,
) -> ReportTelemetryArchiveSpoolBudget {
    ReportTelemetryArchiveSpoolBudget {
        closed: budget.closed,
        finalizing: budget.finalizing,
        accounted_bytes: budget.accounted_bytes,
        accounted_files: budget.accounted_files,
        ordinary_growth_bytes: budget.ordinary_growth_bytes,
        ordinary_growth_files: budget.ordinary_growth_files,
        control_growth_bytes: budget.control_growth_bytes,
        control_growth_files: budget.control_growth_files,
        ordinary_frames: budget.ordinary_frames,
        control_frames: budget.control_frames,
        outstanding_leases: budget.outstanding_leases,
        protected_reserve_bytes: budget.protected_reserve_bytes,
        protected_reserve_files: budget.protected_reserve_files,
        finalization_reserve_bytes: budget.finalization_reserve_bytes,
        finalization_reserve_files: budget.finalization_reserve_files,
        high_water_bytes: budget.high_water_bytes,
        high_water_files: budget.high_water_files,
    }
}

fn report_loss_range(
    loss: &aiperf_telemetry_archive::ExactLossRangeV1,
) -> ReportTelemetryLossRange {
    ReportTelemetryLossRange {
        source_id: loss.source_id.clone(),
        loss_kind: report_loss_kind(loss.loss_kind),
        reason: report_loss_reason(loss.reason),
        count: loss.count,
        first_source_record_seq: loss.first_source_record_seq,
        last_source_record_seq: loss.last_source_record_seq,
        first_request_attempt_seq: loss.first_request_attempt_seq,
        last_request_attempt_seq: loss.last_request_attempt_seq,
        first_tick: loss.first_tick,
        last_tick: loss.last_tick,
        first_deadline_ns: loss.first_deadline_ns,
        last_deadline_ns: loss.last_deadline_ns,
        loss_observed_ns: loss.loss_observed_ns,
        boundary_refs: loss
            .boundary_refs
            .iter()
            .map(report_boundary_reference)
            .collect(),
        boundary_overflow_count: loss.boundary_overflow_count,
        boundary_overflow_digest: loss.boundary_overflow_digest.map(Digest::to_tagged_hex),
    }
}

fn report_loss_saturation(
    loss: &aiperf_telemetry_archive::LossSaturationSnapshotV1,
) -> ReportTelemetryLossSaturationSummary {
    ReportTelemetryLossSaturationSummary {
        source_id: loss.source_id.clone(),
        loss_kind: report_loss_kind(loss.loss_kind),
        reason: report_loss_reason(loss.reason),
        saturation_slot_id: loss.saturation_slot_id.to_tagged_hex(),
        saturation_snapshot_seq: loss.saturation_snapshot_seq,
        cumulative_omitted_range_count: loss.cumulative_omitted_range_count,
        cumulative_omitted_entry_count: loss.cumulative_omitted_entry_count,
        omitted_rolling_digest: loss.omitted_rolling_digest.to_tagged_hex(),
        first_source_record_seq: loss.first_source_record_seq,
        last_source_record_seq: loss.last_source_record_seq,
        first_request_attempt_seq: loss.first_request_attempt_seq,
        last_request_attempt_seq: loss.last_request_attempt_seq,
        first_tick: loss.first_tick,
        last_tick: loss.last_tick,
        first_deadline_ns: loss.first_deadline_ns,
        last_deadline_ns: loss.last_deadline_ns,
        loss_observed_ns: loss.loss_observed_ns,
    }
}

fn report_boundary_reference(
    boundary: &aiperf_telemetry_archive::BoundaryReference,
) -> ReportTelemetryBoundaryReference {
    ReportTelemetryBoundaryReference {
        transition_id: boundary.transition_id.clone(),
        boundary_id: boundary.boundary_id.clone(),
        phase_id: boundary.phase_id.clone(),
        source_id: boundary.source_id.clone(),
        role: match boundary.role {
            aiperf_telemetry_archive::BoundaryRole::PhaseStart => {
                ReportTelemetryBoundaryRole::PhaseStart
            }
            aiperf_telemetry_archive::BoundaryRole::PhaseEnd => {
                ReportTelemetryBoundaryRole::PhaseEnd
            }
        },
        coalescing_group_id: boundary.coalescing_group_id.clone(),
    }
}

const fn report_loss_kind(kind: aiperf_telemetry_archive::LossKindV1) -> ReportTelemetryLossKind {
    match kind {
        aiperf_telemetry_archive::LossKindV1::MissedCadence => {
            ReportTelemetryLossKind::MissedCadence
        }
        aiperf_telemetry_archive::LossKindV1::ArchiveRejected => {
            ReportTelemetryLossKind::ArchiveRejected
        }
        aiperf_telemetry_archive::LossKindV1::ProjectionFailed => {
            ReportTelemetryLossKind::ProjectionFailed
        }
        aiperf_telemetry_archive::LossKindV1::WriterFailed => ReportTelemetryLossKind::WriterFailed,
        aiperf_telemetry_archive::LossKindV1::ShutdownAbandoned => {
            ReportTelemetryLossKind::ShutdownAbandoned
        }
    }
}

const fn report_loss_reason(
    reason: aiperf_telemetry_archive::LossReasonV1,
) -> ReportTelemetryLossReason {
    match reason {
        aiperf_telemetry_archive::LossReasonV1::CadenceOverrun => {
            ReportTelemetryLossReason::CadenceOverrun
        }
        aiperf_telemetry_archive::LossReasonV1::ArchiveAdmissionRejected => {
            ReportTelemetryLossReason::ArchiveAdmissionRejected
        }
        aiperf_telemetry_archive::LossReasonV1::ProjectionError => {
            ReportTelemetryLossReason::ProjectionError
        }
        aiperf_telemetry_archive::LossReasonV1::WriterError => {
            ReportTelemetryLossReason::WriterError
        }
        aiperf_telemetry_archive::LossReasonV1::ShutdownDeadline => {
            ReportTelemetryLossReason::ShutdownDeadline
        }
    }
}

fn prepared_outcome(
    run_id: String,
    archive: ReportTelemetryArchive,
    driver_attempts: u64,
) -> Result<PreparedRunOutcome> {
    let outcome = RunOutcome {
        run: ReportRunInfo {
            mode: Some(WORKLOAD_ID.to_owned()),
            model: None,
        },
        telemetry_archive: Some(archive.clone()),
        ..RunOutcome::default()
    };
    let native_report = NativeReport::from_input(NativeReportInput {
        metrics: None,
        outcome: &outcome,
    });
    let mut provenance = BTreeMap::new();
    provenance.insert("telemetry_archive_id".to_owned(), archive.archive_id);
    provenance.insert(
        "telemetry_archive_state".to_owned(),
        if archive.finalized_remote {
            "remotely_finalized"
        } else {
            "locally_finalized"
        }
        .to_owned(),
    );
    provenance.insert("telemetry_run_id".to_owned(), run_id);
    provenance.insert(
        "telemetry_source_attempts".to_owned(),
        driver_attempts.to_string(),
    );
    Ok(PreparedRunOutcome {
        native_report,
        report_facts: ReportPairRunFacts::new(),
        provenance,
        report_commit: None,
    })
}

async fn wait_for_stop(
    clock: Rc<dyn Clock>,
    start_ns: i64,
    duration_ns: Option<i64>,
) -> Result<TerminationReason> {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        let mut interrupt = signal(SignalKind::interrupt())?;
        let mut terminate = signal(SignalKind::terminate())?;
        if let Some(duration_ns) = duration_ns {
            let deadline = checked_deadline(start_ns, duration_ns)?;
            let remaining = deadline.saturating_sub(clock.now_ns());
            let sleep = clock.clone().sleep(remaining);
            tokio::pin!(sleep);
            tokio::select! {
                () = &mut sleep => Ok(TerminationReason::Duration),
                _ = interrupt.recv() => Ok(TerminationReason::Signal),
                _ = terminate.recv() => Ok(TerminationReason::Signal),
            }
        } else {
            tokio::select! {
                _ = interrupt.recv() => Ok(TerminationReason::Signal),
                _ = terminate.recv() => Ok(TerminationReason::Signal),
            }
        }
    }
    #[cfg(not(unix))]
    {
        match duration_ns {
            Some(duration_ns) => {
                let deadline = checked_deadline(start_ns, duration_ns)?;
                clock
                    .clone()
                    .sleep(deadline.saturating_sub(clock.now_ns()))
                    .await;
                Ok(TerminationReason::Duration)
            }
            None => {
                tokio::signal::ctrl_c().await?;
                Ok(TerminationReason::Signal)
            }
        }
    }
}

async fn await_before<F, T>(clock: Rc<dyn Clock>, deadline_ns: i64, future: F) -> Result<T>
where
    F: std::future::Future<Output = T>,
{
    let remaining = deadline_ns.saturating_sub(clock.now_ns());
    ensure!(remaining > 0, "telemetry shutdown deadline expired");
    let sleep = clock.clone().sleep(remaining);
    tokio::pin!(sleep);
    tokio::pin!(future);
    tokio::select! {
        biased;
        output = &mut future => Ok(output),
        () = &mut sleep => bail!("telemetry shutdown deadline expired"),
    }
}

fn receipt_epoch(
    execution_id: ExecutionId,
    session_id: Option<SessionId>,
    anchor: EpochAnchor,
    distribution_id: Digest,
) -> Result<ReceiptObserverEpochV1> {
    Ok(ReceiptObserverEpochV1::new(
        execution_id,
        session_id,
        TimeDomain::Real,
        anchor.clock_ns,
        Some(anchor.unix_epoch_ns),
        anchor.capture_uncertainty_ns,
        distribution_id,
    )?)
}

fn source_descriptors(sources: &[ValidatedTelemetrySourceV2]) -> Result<CanonicalJsonValue> {
    let values =
        sources
            .iter()
            .map(|source| {
                let attributes =
                    CanonicalJsonValue::object(source.attributes.iter().map(|(key, value)| {
                        (key.clone(), CanonicalJsonValue::String(value.clone()))
                    }))?;
                CanonicalJsonValue::object([
                    ("attributes".to_owned(), attributes),
                    ("config".to_owned(), source.persistent_identity.clone()),
                    (
                        "id".to_owned(),
                        CanonicalJsonValue::String(source.id.clone()),
                    ),
                    (
                        "interval_ns".to_owned(),
                        CanonicalJsonValue::Integer(i128::from(source.interval_ns)),
                    ),
                    (
                        "request_timeout_ns".to_owned(),
                        CanonicalJsonValue::Integer(i128::from(source.request_timeout_ns)),
                    ),
                    (
                        "type".to_owned(),
                        CanonicalJsonValue::String(source.source_type.clone()),
                    ),
                ])
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;
    Ok(CanonicalJsonValue::Array(values))
}

fn collect_identity_digest(
    archive: &PreparedTelemetryArchiveCollectComponents,
    sources: &CanonicalJsonValue,
) -> Result<Digest> {
    let parser_role_matrix =
        CanonicalJsonValue::parse_canonical(aiperf_prometheus::role_validity_matrix_v1_bytes())
            .context("validating the frozen telemetry parser role matrix")?;
    let components = CanonicalJsonValue::Array(
        archive
            .persistent_component_identities
            .iter()
            .map(|identity| identity.canonical_descriptor())
            .collect(),
    );
    let identity = CanonicalJsonValue::object([
        ("components".to_owned(), components),
        (
            "parser_id".to_owned(),
            CanonicalJsonValue::String("strict_exposition_v1".to_owned()),
        ),
        ("parser_role_validity_matrix".to_owned(), parser_role_matrix),
        (
            "required".to_owned(),
            CanonicalJsonValue::Bool(archive.required),
        ),
        ("sources".to_owned(), sources.clone()),
        (
            "spool_quota_bytes".to_owned(),
            CanonicalJsonValue::Integer(i128::from(archive.spool_quota_bytes)),
        ),
        (
            "spool_quota_files".to_owned(),
            CanonicalJsonValue::Integer(i128::from(archive.spool_quota_files)),
        ),
        (
            "target".to_owned(),
            CanonicalJsonValue::String(archive.target.as_str().to_owned()),
        ),
        (
            "writer".to_owned(),
            archive.writer.persistent_writer_identity().clone(),
        ),
    ])?;
    Ok(domain_digest(
        "aiperf.archive.config.v1",
        &[identity.to_bytes().as_slice()],
    ))
}

fn prepare_spool_budget(
    target_partition_bytes: u64,
    raw_max_plaintext_bytes: Option<u64>,
    spool_quota_bytes: u64,
    spool_quota_files: u64,
    sources: &[ValidatedTelemetrySourceV2],
    spool: &QualifiedSpool,
    ordinary_frame_capacity: usize,
    control_frame_capacity: usize,
) -> Result<(
    Arc<dyn ArchiveSpoolBudgetAuthority>,
    ArchiveProjectionFootprint,
    ArchiveProjectionFootprint,
)> {
    let maximum_encoded_bytes = sources
        .iter()
        .map(|source| u64::try_from(source.prepared.maximum_encoded_entity_bytes()))
        .collect::<std::result::Result<Vec<_>, _>>()?
        .into_iter()
        .max()
        .ok_or_else(|| anyhow!("telemetry collection has no source entity bound"))?;
    let maximum_decoded_bytes = sources
        .iter()
        .map(|source| u64::try_from(source.prepared.maximum_decoded_entity_bytes()))
        .collect::<std::result::Result<Vec<_>, _>>()?
        .into_iter()
        .max()
        .ok_or_else(|| anyhow!("telemetry collection has no source projection bound"))?;
    let ordinary_bytes = maximum_decoded_bytes
        .checked_mul(SOURCE_TO_WAL_EXPANSION_BOUND)
        .and_then(|bytes| bytes.checked_add(maximum_encoded_bytes))
        .and_then(|bytes| bytes.checked_add(8 * MIB))
        .ok_or_else(|| anyhow!("telemetry source projection bound overflowed"))?;
    ensure!(
        ordinary_bytes
            .checked_add(MIB)
            .is_some_and(|bytes| bytes <= DEFAULT_MAX_WAL_FRAME_BYTES),
        "telemetry source limits cannot fit one complete WAL frame"
    );
    let ordinary_projection_footprint = ArchiveProjectionFootprint {
        bytes: ordinary_bytes,
        frames: 1,
        files: 16,
    };
    let control_projection_footprint = ArchiveProjectionFootprint {
        bytes: 2 * MIB,
        frames: 1,
        files: 8,
    };
    let open_parquet_bytes = target_partition_bytes
        .checked_mul(ARCHIVE_TABLE_COUNT)
        .and_then(|bytes| bytes.checked_mul(2))
        .ok_or_else(|| anyhow!("telemetry open-Parquet reserve overflowed"))?;
    let fallback_wal_bytes = target_partition_bytes
        .checked_add(ordinary_bytes)
        .ok_or_else(|| anyhow!("telemetry fallback-WAL reserve overflowed"))?;
    let control_lane_bytes = control_projection_footprint
        .bytes
        .checked_mul(u64::try_from(control_frame_capacity)?)
        .ok_or_else(|| anyhow!("telemetry control-lane reserve overflowed"))?;
    let control_lane_files = control_projection_footprint
        .files
        .checked_mul(u64::try_from(control_frame_capacity)?)
        .ok_or_else(|| anyhow!("telemetry control-lane file reserve overflowed"))?;
    let optional_raw_object = raw_max_plaintext_bytes.map_or(
        Ok(ArchiveSpoolResources::default()),
        |max_plaintext_bytes| -> Result<ArchiveSpoolResources> {
            Ok(ArchiveSpoolResources {
                bytes: max_plaintext_bytes
                    .checked_add(MIB)
                    .ok_or_else(|| anyhow!("telemetry raw-object reserve overflowed"))?,
                files: 2,
            })
        },
    )?;
    let limits = ArchiveSpoolBudgetLimits {
        quota: ArchiveSpoolResources {
            bytes: spool_quota_bytes,
            files: spool_quota_files,
        },
        ordinary_frame_capacity: u64::try_from(ordinary_frame_capacity)?,
        control_frame_capacity: u64::try_from(control_frame_capacity)?,
        reserve: ArchiveSpoolReservePlan {
            largest_wal_frame: ArchiveSpoolResources {
                bytes: ordinary_bytes + MIB,
                files: 2,
            },
            fallback_wal_window: ArchiveSpoolResources {
                bytes: fallback_wal_bytes,
                files: 4,
            },
            open_parquet_builders: ArchiveSpoolResources {
                bytes: open_parquet_bytes,
                files: ARCHIVE_TABLE_COUNT * 2,
            },
            cow_index_path: ArchiveSpoolResources {
                bytes: INDEX_PATH_PAGE_BOUND * MIB,
                files: INDEX_PATH_PAGE_BOUND * 2,
            },
            generation_and_head: ArchiveSpoolResources {
                bytes: 8 * MIB,
                files: 8,
            },
            receipt_transaction: ArchiveSpoolResources {
                bytes: 8 * MIB,
                files: 12,
            },
            optional_raw_object,
            wal_seal: ArchiveSpoolResources {
                bytes: MIB,
                files: 4,
            },
            emergency_finalization: ArchiveSpoolResources {
                bytes: 2 * MIB,
                files: 8,
            },
            control_lane: ArchiveSpoolResources {
                bytes: control_lane_bytes,
                files: control_lane_files,
            },
        },
    };
    let authority = AtomicArchiveSpoolBudget::new(limits, spool.budget_observation()?)?;
    let authority: Arc<dyn ArchiveSpoolBudgetAuthority> = authority;
    Ok((
        authority,
        ordinary_projection_footprint,
        control_projection_footprint,
    ))
}

fn prepare_artifact_target(path: &std::path::Path) -> Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| anyhow!("telemetry artifact target has no parent"))?;
    ensure!(
        parent.is_dir(),
        "telemetry artifact target parent does not exist or is not a directory"
    );
    std::fs::create_dir(path).with_context(|| {
        format!(
            "creating exclusive telemetry artifact target {}",
            path.display()
        )
    })?;
    Ok(())
}

#[derive(Serialize)]
#[serde(deny_unknown_fields)]
struct ArchiveFailureDiagnosticV1 {
    schema_version: &'static str,
    kind: &'static str,
    stage: &'static str,
    code: String,
    message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    archive_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    local_head: Option<ArchiveFailureHeadV1>,
}

#[derive(Serialize)]
#[serde(deny_unknown_fields)]
struct ArchiveFailureHeadV1 {
    generation_key: String,
    generation_hash: String,
    index_root_key: String,
    index_root_hash: String,
    archive_state: &'static str,
}

fn archive_reporting_failure(
    artifact_target: &std::path::Path,
    code: &str,
    message: impl Into<String>,
    archive_id: Option<ArchiveId>,
    local_head: Option<&aiperf_telemetry_archive::HeadDescriptorV1>,
) -> anyhow::Error {
    let message = redact_diagnostic(message.into());
    let artifact =
        write_archive_failure_diagnostic(artifact_target, code, &message, archive_id, local_head);
    let failure = match artifact {
        Ok(artifact) => PreparedRunFailure::reporting(code, message.clone())
            .and_then(|failure| failure.with_diagnostic_artifacts(vec![artifact])),
        Err(error) => PreparedRunFailure::reporting(
            "archive_diagnostic_persistence_failed",
            format!("{message}; diagnostic persistence failed: {error:#}"),
        ),
    };
    match failure {
        Ok(failure) => anyhow::Error::new(failure),
        Err(error) => error,
    }
}

fn write_archive_failure_diagnostic(
    artifact_target: &std::path::Path,
    code: &str,
    message: &str,
    archive_id: Option<ArchiveId>,
    local_head: Option<&aiperf_telemetry_archive::HeadDescriptorV1>,
) -> Result<RunDiagnosticArtifactV2> {
    ensure!(
        artifact_target.is_dir(),
        "diagnostic artifact target is absent"
    );
    let diagnostic = ArchiveFailureDiagnosticV1 {
        schema_version: "1.0",
        kind: "archive_failure_diagnostic",
        stage: "reporting",
        code: code.to_owned(),
        message: message.to_owned(),
        archive_id: archive_id.map(uuid_archive),
        local_head: local_head.map(|head| ArchiveFailureHeadV1 {
            generation_key: head.generation_key.clone(),
            generation_hash: head.generation_hash.to_tagged_hex(),
            index_root_key: head.index_root_key.clone(),
            index_root_hash: head.index_root_hash.to_tagged_hex(),
            archive_state: archive_state_name(head.archive_state),
        }),
    };
    let mut bytes =
        serde_json::to_vec_pretty(&diagnostic).context("serializing archive failure diagnostic")?;
    bytes.push(b'\n');
    let content_hash = format!("blake3:{}", blake3::hash(&bytes).to_hex());
    let temporary_path = artifact_target.join(".archive-failure-diagnostic.json.tmp");
    let final_path = artifact_target.join(ARCHIVE_FAILURE_DIAGNOSTIC_PATH);
    let mut file = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&temporary_path)
        .with_context(|| {
            format!(
                "creating archive failure diagnostic temporary file {}",
                temporary_path.display()
            )
        })?;
    file.write_all(&bytes)
        .context("writing archive failure diagnostic")?;
    file.sync_all()
        .context("syncing archive failure diagnostic")?;
    drop(file);
    std::fs::rename(&temporary_path, &final_path)
        .context("committing archive failure diagnostic")?;
    std::fs::File::open(artifact_target)
        .and_then(|directory| directory.sync_all())
        .context("syncing archive diagnostic directory")?;
    Ok(RunDiagnosticArtifactV2 {
        kind: "archive_failure_diagnostic".to_owned(),
        relative_path: std::path::PathBuf::from(ARCHIVE_FAILURE_DIAGNOSTIC_PATH),
        content_hash,
    })
}

const fn archive_state_name(state: ArchiveState) -> &'static str {
    match state {
        ArchiveState::Open => "open",
        ArchiveState::StopRequested => "stop_requested",
        ArchiveState::LocallyFinalized => "locally_finalized",
        ArchiveState::RemotelyFinalized => "remotely_finalized",
        ArchiveState::Failed => "failed",
    }
}

fn local_report_head(
    spool: &std::path::Path,
    head: &aiperf_telemetry_archive::HeadDescriptorV1,
) -> Result<ReportTelemetryArchiveHead> {
    Ok(ReportTelemetryArchiveHead {
        head_uri: local_file_uri(&spool.join("LOCAL-LATEST"))?,
        generation_uri: local_file_uri(&spool.join(&head.generation_key))?,
        generation_hash: head.generation_hash.to_tagged_hex(),
        index_root_hash: head.index_root_hash.to_tagged_hex(),
    })
}

fn remote_report_head(
    target: &crate::telemetry_watch::NormalizedArchiveUri,
    head: &aiperf_telemetry_archive::HeadDescriptorV1,
) -> ReportTelemetryArchiveHead {
    ReportTelemetryArchiveHead {
        head_uri: join_archive_uri(target, "LATEST"),
        generation_uri: join_archive_uri(target, &head.generation_key),
        generation_hash: head.generation_hash.to_tagged_hex(),
        index_root_hash: head.index_root_hash.to_tagged_hex(),
    }
}

fn local_file_uri(path: &std::path::Path) -> Result<String> {
    url::Url::from_file_path(path)
        .map(|url| url.to_string())
        .map_err(|()| anyhow!("local archive path cannot be represented as a file URI"))
}

fn join_archive_uri(target: &crate::telemetry_watch::NormalizedArchiveUri, key: &str) -> String {
    format!("{}/{}", target.as_str().trim_end_matches('/'), key)
}

fn checked_deadline(start_ns: i64, duration_ns: i64) -> Result<i64> {
    start_ns
        .checked_add(duration_ns)
        .ok_or_else(|| anyhow!("telemetry Clock deadline overflowed"))
}

fn parse_distribution_digest(value: &str) -> Result<Digest> {
    Digest::parse(value.strip_prefix("blake3:").unwrap_or(value))
        .map_err(|error| anyhow!("invalid runner distribution digest: {error}"))
}

fn archive_id(value: Uuid) -> Result<ArchiveId> {
    ArchiveId::new(*value.as_bytes()).map_err(anyhow::Error::from)
}

fn session_id(value: Uuid) -> Result<SessionId> {
    SessionId::new(*value.as_bytes()).map_err(anyhow::Error::from)
}

fn uuid_archive(value: ArchiveId) -> String {
    Uuid::from_bytes(*value.as_bytes()).to_string()
}

fn uuid_session(value: SessionId) -> String {
    Uuid::from_bytes(*value.as_bytes()).to_string()
}

fn telemetry_workload(
    workload: &dyn ValidatedWorkloadConfig,
) -> Result<&ValidatedTelemetryWatchWorkloadV2> {
    workload
        .as_any()
        .downcast_ref::<ValidatedTelemetryWatchWorkloadV2>()
        .ok_or_else(|| anyhow!("telemetry watch received another workload config"))
}

fn component_error(error: ArchiveComponentError) -> anyhow::Error {
    anyhow!(error.to_string())
}
