// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Feature-gated in-process co-simulation against Dynamo's passive mock engine.
//!
//! This module is the application-side realization of
//! `specs/2026-07-10-steppable-clock-injected-engine-design.md`: AIPerf owns the
//! [`SimClock`], workload loop, observers, and report; `dynamo-mocker` owns only
//! scheduler/performance-model state behind [`SteppableReplay`]. There are no
//! sockets, subprocesses, shared-memory rings, or secondary clocks.
//!
//! The dependency is intentionally compiled only with `dynosim`. The
//! normal HTTP binary and every library crate below this application layer stay
//! independent of Dynamo.

use std::cell::{Cell, RefCell};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap, HashMap};
use std::future::Future;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::rc::{Rc, Weak};
use std::sync::Arc;
use std::task::{Context as TaskContext, Poll, Waker};

use crate::clock::{Clock, SimClock};
use crate::dataset::{Handle, TextTokenizer, TiktokenTokenizer};
use crate::endpoints::chat_request_body;
use crate::graph::bench::{BenchConfig, build_workload};
use crate::graph::execution::{LocalGraphTraceExecutionBackend, TracePlacement};
use crate::graph::executor::{ExecutorFlags, TraceExecutor};
use crate::graph::materialize::SegmentItemsMaterializer;
use crate::graph::model::{GraphTracePlan, TraceRecord};
use crate::graph::policy::{
    CompositeNodeDispatchPolicy, NodeDispatchPolicy, NodeFailurePolicy, PrefillSlotNodePolicy,
};
use crate::graph::runtime::{
    SimDriveError, SimEventSource, SimStep, drive_real_with_source, drive_sim_with_source,
};
use crate::graph::segment::SegmentStore;
use crate::graph::sink::{GraphDispatchOptions, GraphReply, GraphSink};
use crate::graph::transport_bench::GraphRpsReport;
use crate::graph::wire::OpenAiChatMessage as GraphMessage;
use crate::graph::workload::{GraphWorkload, GraphWorkloadReport};
use crate::metrics_core::{
    AccumulatorSummary, ExportContext, InferenceDimensions, MetricTag, MetricsAccumulator,
    MetricsConfig, Phase as MetricsPhase, RecordIngest, RequestTrace,
};
use crate::timing::{ArrivalPattern, Phase, PhaseStats, SlotPool, StopConfig};
use anyhow::{Context, Result};
use async_trait::async_trait;
use bytes::Bytes;
use dynamo_mocker::common::protocols::{DirectRequest, EngineType, MockEngineArgs, WorkerType};
use dynamo_mocker::loadgen::{
    AgenticTrace, DynamoRequestTrace, EngineEvent, SteppableAgg, SteppableDisagg, SteppableEngine,
    SteppableReplay, Trace, TraceFileFormat, TurnTrace, WorkloadDriver,
};
use dynamo_mocker::replay::{
    OfflineDisaggReplayConfig, OfflineTraceReplayConfig as DynamoTraceReplayConfig,
    OfflineTraceReplayEngines, ReplayKvEventVisibility, ReplayKvRouterConfig,
    ReplayPrefillLoadEstimator, ReplayRouterMode, ReplayTerminalStatus as DynamoTerminalStatus,
    SlaThresholds as DynamoSlaThresholds, TraceDistributionStats as DynamoDistributionStats,
    TraceSimulationReport as DynamoSimulationReport, generate_trace_worker_artifacts_offline,
    generate_trace_worker_artifacts_offline_with_kv_event_visibility,
    simulate_concurrency_live_requests, simulate_offline_trace_files,
};
use loadgen_core::collector::ReplayTerminalStatus;
use loadgen_core::observer::CollectorObserver;
use loadgen_core::sink::{ObservedTokenKind, ObservedUsage, RequestObserver, RequestSink};
use rustc_hash::FxHashMap;
use tokio::sync::Notify;
use uuid::Uuid;

use crate::adaptive::AdaptiveRunConfig;
use crate::ancillary::AncillaryTimingConfig;
use crate::fixed_schedule::{
    DatasetFixedScheduleSource, FixedScheduleConfig, FixedScheduleWorkload,
};
use crate::metrics::{
    NativeMetricsObserver, NativeResponseMetadata, ObserverTee, RequestMetricMetadata,
};
use crate::multiturn::{ConversationSource, TurnToSend};
use crate::phase_runtime::{PhasedScheduledRunReport, ScheduledPhaseReport};
use crate::request_rate::RequestRateConfig;
use crate::run::{
    OnlineRunReport, run_paced_with_backend, run_request_rate_with_backend,
    run_user_centric_adaptive_with_backend, run_user_centric_with_backend,
};
use crate::scheduled::{
    ModelResponseMetadata, ScheduledAncillaryPolicies, ScheduledRunReport,
    SingleTurnDatasetWorkload, TurnDispatchOutcome, TurnDispatcher, TurnRecordProcessor, Workload,
    run_scheduled_workload_with_processors,
};
use crate::transport::core::Request;
use crate::transport::http::{HttpDispatchResult, HttpRequestDispatcher};
use crate::user_centric::UserCentricConfig;
use crate::workload::SkeletonWorkload;

const NS_PER_MS: f64 = 1_000_000.0;

/// Deployment topology simulated by Dynamo.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OfflineTopology {
    /// One scheduler/engine worker without a router.
    Single,
    /// Multiple aggregate workers behind one request router.
    Aggregated,
    /// Separate prefill and decode worker pools.
    Disaggregated,
}

/// Router policy for aggregate and disaggregated topologies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OfflineRouterMode {
    /// Stable worker rotation; deterministic when all other inputs are fixed.
    RoundRobin,
    /// Dynamo's prefix-affinity/load-aware KV router.
    Kv,
}

/// Construction seam for an in-process offline replay engine.
///
/// The workload, clock pump, observers, and reports depend only on the
/// [`SteppableReplay`] contract. A different simulator can therefore replace
/// Dynamo's native constructor without changing any scheduling policy.
pub trait OfflineEngineFactory {
    /// Build one passive engine from the validated frontend configuration.
    fn build(&self, config: &OfflineEngineConfig) -> Result<Box<dyn SteppableReplay>>;
}

/// Native Dynamo implementation of [`OfflineEngineFactory`].
#[derive(Debug, Clone, Copy, Default)]
pub struct NativeDynamoEngineFactory;

/// Canonical Dynamo trace input format accepted by native AIPerf offline runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OfflineTraceFormat {
    /// Mooncake request/cache trace with cumulative prompt rows.
    Mooncake,
    /// Mooncake-shaped rows whose later inputs are session deltas.
    MooncakeDelta,
    /// Request-level workflow DAG with explicit dependencies.
    AgenticMooncake,
    /// Applied Compute agentic session trace.
    AppliedComputeAgentic,
    /// One or more native Dynamo request-trace files.
    Dynamo,
}

impl From<OfflineTraceFormat> for TraceFileFormat {
    fn from(value: OfflineTraceFormat) -> Self {
        match value {
            OfflineTraceFormat::Mooncake => Self::Mooncake,
            OfflineTraceFormat::MooncakeDelta => Self::MooncakeDelta,
            OfflineTraceFormat::AgenticMooncake => Self::AgenticMooncake,
            OfflineTraceFormat::AppliedComputeAgentic => Self::AppliedComputeAgentic,
            OfflineTraceFormat::Dynamo => Self::Dynamo,
        }
    }
}

/// Native trace-replay workload controls shared with canonical DynoSim replay.
#[derive(Debug, Clone)]
pub struct OfflineTraceConfig {
    /// Trace files; only native Dynamo format accepts more than one.
    pub paths: Vec<PathBuf>,
    /// Parser/lowering format.
    pub format: OfflineTraceFormat,
    /// Tokens represented by each trace hash. Native Dynamo traces derive this
    /// from their header when omitted.
    pub trace_block_size: Option<usize>,
    /// Optional depth-first active-session cap.
    pub replay_concurrency: Option<usize>,
    /// Divide authored arrivals and inter-turn delays by this factor.
    pub arrival_speedup_ratio: f64,
    /// Shared initial-prefix fraction for Applied Compute agentic traces.
    pub shared_prefix_ratio: f64,
    /// Shared-prefix group count for Applied Compute agentic traces.
    pub num_prefix_groups: usize,
    /// Optional soft cap on simulated time in milliseconds.
    pub max_sim_time_ms: Option<f64>,
}

/// Optional visibility override for timed KV events in worker artifacts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OfflineKvEventVisibility {
    /// Publish cache mutations at scheduler-pass start.
    PassStart,
    /// Publish cache mutations at scheduler-pass completion.
    PassEnd,
}

impl From<OfflineKvEventVisibility> for ReplayKvEventVisibility {
    fn from(value: OfflineKvEventVisibility) -> Self {
        match value {
            OfflineKvEventVisibility::PassStart => Self::PassStart,
            OfflineKvEventVisibility::PassEnd => Self::PassEnd,
        }
    }
}

/// Canonical replay-level AIC overrides applied to every engine role.
#[derive(Debug, Clone, Default)]
pub struct OfflineAicConfig {
    /// Serving backend identity.
    pub backend: Option<String>,
    /// GPU system identity.
    pub system: Option<String>,
    /// Backend version in the AIC performance database.
    pub backend_version: Option<String>,
    /// Tensor-parallel degree.
    pub tp_size: Option<usize>,
    /// Hugging Face model path used by AIC.
    pub model_path: Option<String>,
    /// MoE tensor-parallel degree.
    pub moe_tp_size: Option<usize>,
    /// MoE expert-parallel degree.
    pub moe_ep_size: Option<usize>,
    /// Attention data-parallel degree.
    pub attention_dp_size: Option<usize>,
    /// GEMM quantization override.
    pub gemm_dtype: Option<String>,
    /// MoE quantization override.
    pub moe_dtype: Option<String>,
    /// FMHA quantization override.
    pub fmha_dtype: Option<String>,
    /// KV-cache quantization override.
    pub kv_cache_dtype: Option<String>,
    /// Communication quantization override.
    pub comm_dtype: Option<String>,
    /// Speculative draft-token count.
    pub nextn: Option<usize>,
    /// Comma-separated conditional draft acceptance rates.
    pub nextn_accept_rates: Option<String>,
}

impl OfflineAicConfig {
    fn requested(&self) -> bool {
        self.backend.is_some()
            || self.system.is_some()
            || self.backend_version.is_some()
            || self.tp_size.is_some()
            || self.model_path.is_some()
            || self.moe_tp_size.is_some()
            || self.moe_ep_size.is_some()
            || self.attention_dp_size.is_some()
            || self.gemm_dtype.is_some()
            || self.moe_dtype.is_some()
            || self.fmha_dtype.is_some()
            || self.kv_cache_dtype.is_some()
            || self.comm_dtype.is_some()
            || self.nextn.is_some()
            || self.nextn_accept_rates.is_some()
    }

    fn apply(&self, args: &mut MockEngineArgs) -> Result<()> {
        if !self.requested() {
            return Ok(());
        }
        let backend = self
            .backend
            .clone()
            .context("AIC replay modeling requires --aic-backend")?;
        let system = self
            .system
            .clone()
            .context("AIC replay modeling requires --aic-system")?;
        let model_path = self
            .model_path
            .clone()
            .context("AIC replay modeling requires --aic-model-path")?;
        args.aic_backend = Some(backend);
        args.aic_system = Some(system);
        args.aic_model_path = Some(model_path);
        args.aic_backend_version = self.backend_version.clone();
        args.aic_tp_size = Some(self.tp_size.unwrap_or(1));
        args.aic_moe_tp_size = self.moe_tp_size;
        args.aic_moe_ep_size = self.moe_ep_size;
        args.aic_attention_dp_size = self.attention_dp_size;
        args.aic_gemm_dtype = self.gemm_dtype.clone();
        args.aic_moe_dtype = self.moe_dtype.clone();
        args.aic_fmha_dtype = self.fmha_dtype.clone();
        args.aic_kv_cache_dtype = self.kv_cache_dtype.clone();
        args.aic_comm_dtype = self.comm_dtype.clone();
        args.aic_nextn = self.nextn;
        args.aic_nextn_accept_rates = self.nextn_accept_rates.clone();
        Ok(())
    }
}

impl From<OfflineRouterMode> for ReplayRouterMode {
    fn from(value: OfflineRouterMode) -> Self {
        match value {
            OfflineRouterMode::RoundRobin => Self::RoundRobin,
            OfflineRouterMode::Kv => Self::KvRouter,
        }
    }
}

/// Dynamo engine construction selected by the feature-gated CLI.
#[derive(Debug, Clone)]
pub struct OfflineEngineConfig {
    /// Optional JSON profile accepted by `MockEngineArgs::from_json_file`.
    pub profile: Option<PathBuf>,
    /// Inline JSON accepted by Dynamo's canonical `--extra-engine-args`.
    pub extra_engine_args: Option<String>,
    /// Inline JSON for a disaggregated prefill engine profile.
    pub prefill_engine_args: Option<String>,
    /// Inline JSON for a disaggregated decode engine profile.
    pub decode_engine_args: Option<String>,
    /// Complete inline `KvRouterConfig` JSON.
    pub router_config: Option<String>,
    /// Startup-only router policy-family YAML path, overriding the JSON field.
    pub router_policy_config: Option<PathBuf>,
    /// Model selector for a multi-model router policy document.
    pub router_model_name: Option<String>,
    /// Replay-level AIC overrides shared by aggregate/prefill/decode roles.
    pub aic: Option<OfflineAicConfig>,
    /// Retain complete backend per-request records for raw export.
    pub capture_per_request: bool,
    /// Canonical goodput thresholds evaluated by the Dynamo collector.
    pub sla: DynamoSlaThresholds,
    /// Deployment topology.
    pub topology: OfflineTopology,
    /// Aggregate worker count.
    pub workers: usize,
    /// Disaggregated prefill-worker count.
    pub prefill_workers: usize,
    /// Disaggregated decode-worker count.
    pub decode_workers: usize,
    /// Router policy for routed topologies.
    pub router_mode: OfflineRouterMode,
    /// Drive a single worker with Dynamo's `execute_pass` engine
    /// ([`SteppableEngine`]) instead of the `AggRuntime` (`SteppableAgg`).
    ///
    /// This matches Dynamo's own offline replay, which uses the single-runtime
    /// (`single.rs`, `execute_pass`) whenever `num_workers == 1 && !KvRouter`.
    /// `execute_pass` prefills a request admitted mid-run at its arrival instant,
    /// whereas `AggRuntime`'s dynamic admission defers that prefill to the next
    /// batch boundary — a divergence that only surfaces under prefill saturation.
    /// Selecting the same engine as Dynamo makes offline replay byte-exact at any
    /// concurrency.
    ///
    /// `execute_pass` cannot stop at a finite deadline, so it is only valid when
    /// the run schedules no clock events during engine processing (closed-loop
    /// concurrency with no arrival/cancellation/ramp timing). The run path sets
    /// this only for eligible workloads; it is ignored unless the topology
    /// resolves to a single round-robin worker with no router/estimator/AIC.
    pub single_pass_engine: bool,
}

impl Default for OfflineEngineConfig {
    fn default() -> Self {
        Self {
            profile: None,
            extra_engine_args: None,
            prefill_engine_args: None,
            decode_engine_args: None,
            router_config: None,
            router_policy_config: None,
            router_model_name: None,
            aic: None,
            capture_per_request: false,
            sla: DynamoSlaThresholds::default(),
            topology: OfflineTopology::Single,
            workers: 1,
            prefill_workers: 1,
            decode_workers: 1,
            router_mode: OfflineRouterMode::RoundRobin,
            single_pass_engine: false,
        }
    }
}

impl OfflineEngineConfig {
    /// Set the canonical Dynamo goodput thresholds without exposing the
    /// optional backend crate's concrete DTO to embedding applications.
    ///
    /// The runner uses this narrow constructor so `aiperf` remains the only
    /// crate that directly depends on `dynamo-mocker`. Threshold validation is
    /// completed before an engine is initialized.
    pub fn with_sla_thresholds(
        mut self,
        ttft_ms: Option<f64>,
        itl_ms: Option<f64>,
        e2e_ms: Option<f64>,
    ) -> Result<Self> {
        for (name, value) in [("ttft_ms", ttft_ms), ("itl_ms", itl_ms), ("e2e_ms", e2e_ms)] {
            if let Some(value) = value {
                anyhow::ensure!(
                    value.is_finite() && value >= 0.0,
                    "offline SLA {name} must be finite and non-negative, got {value}"
                );
            }
        }
        self.sla = DynamoSlaThresholds {
            ttft_ms,
            itl_ms,
            e2e_ms,
        };
        Ok(self)
    }

    fn trace_engine_block_size(&self) -> Result<usize> {
        match self.topology {
            OfflineTopology::Disaggregated => {
                let (prefill, _) = self.disaggregated_engine_args()?;
                Ok(prefill.block_size)
            }
            OfflineTopology::Single | OfflineTopology::Aggregated => {
                Ok(self.engine_args()?.block_size)
            }
        }
    }

    fn finalize_engine_args(
        &self,
        mut args: MockEngineArgs,
        label: &str,
    ) -> Result<MockEngineArgs> {
        if let Some(aic) = &self.aic {
            aic.apply(&mut args)?;
        }
        let args = args
            .normalized()
            .with_context(|| format!("invalid Dynamo {label} engine configuration"))?;
        #[cfg(not(feature = "dynamo-kvbm-offload"))]
        anyhow::ensure!(
            args.num_g2_blocks.is_none(),
            "Dynamo KV offload configuration requires the AIPerf `dynamo-kvbm-offload` feature"
        );
        Ok(args)
    }

    #[cfg(not(feature = "dynamo-aic-forward-pass"))]
    fn has_aic_configuration(args: &MockEngineArgs) -> bool {
        args.aic_backend.is_some()
            || args.aic_system.is_some()
            || args.aic_model_path.is_some()
            || args.aic_backend_version.is_some()
            || args.aic_tp_size.is_some()
            || args.aic_moe_tp_size.is_some()
            || args.aic_moe_ep_size.is_some()
            || args.aic_attention_dp_size.is_some()
            || args.aic_gemm_dtype.is_some()
            || args.aic_moe_dtype.is_some()
            || args.aic_fmha_dtype.is_some()
            || args.aic_kv_cache_dtype.is_some()
            || args.aic_comm_dtype.is_some()
            || args.aic_nextn.is_some()
            || args.aic_nextn_accept_rates.is_some()
    }

    fn configure_aic(
        args: MockEngineArgs,
    ) -> Result<(MockEngineArgs, Option<ReplayPrefillLoadEstimator>)> {
        #[cfg(feature = "dynamo-aic-forward-pass")]
        {
            let mut args = args;
            let estimator = crate::aic_runtime::configure_aic_runtime(&mut args)?;
            Ok((args, estimator))
        }
        #[cfg(not(feature = "dynamo-aic-forward-pass"))]
        {
            anyhow::ensure!(
                !Self::has_aic_configuration(&args),
                "AIC engine configuration requires the AIPerf `dynamo-aic-forward-pass` feature"
            );
            Ok((args, None))
        }
    }

    fn validate_prefill_load_estimator(
        &self,
        router_config: Option<&ReplayKvRouterConfig>,
        estimator: Option<ReplayPrefillLoadEstimator>,
    ) -> Result<Option<ReplayPrefillLoadEstimator>> {
        let configured =
            router_config.is_some_and(|config| config.router_prefill_load_model.is_enabled());
        if estimator.is_some() {
            anyhow::ensure!(
                self.router_mode == OfflineRouterMode::Kv,
                "AIC replay modeling requires --offline-router kv"
            );
            anyhow::ensure!(
                configured,
                "AIC replay modeling requires --router-config with router_prefill_load_model='aic'"
            );
        } else {
            anyhow::ensure!(
                !configured,
                "router_prefill_load_model='aic' requires AIC engine configuration"
            );
        }
        Ok(estimator)
    }

    fn engine_args(&self) -> Result<MockEngineArgs> {
        anyhow::ensure!(
            self.profile.is_none() || self.extra_engine_args.is_none(),
            "--engine-profile conflicts with --extra-engine-args"
        );
        match (&self.profile, &self.extra_engine_args) {
            (Some(path), None) => self.finalize_engine_args(
                MockEngineArgs::from_json_file(path).with_context(|| {
                    format!("failed to load Dynamo engine profile {}", path.display())
                })?,
                "aggregate",
            ),
            (None, Some(json)) => self.finalize_engine_args(
                MockEngineArgs::from_json_str(json).context("invalid --extra-engine-args JSON")?,
                "aggregate",
            ),
            (None, None) => self.finalize_engine_args(MockEngineArgs::default(), "default"),
            (Some(_), Some(_)) => unreachable!("conflict checked above"),
        }
    }

    fn disaggregated_engine_args(&self) -> Result<(MockEngineArgs, MockEngineArgs)> {
        match (&self.prefill_engine_args, &self.decode_engine_args) {
            (Some(prefill), Some(decode)) => Ok((
                self.finalize_engine_args(
                    MockEngineArgs::from_json_str(prefill)
                        .context("invalid --prefill-engine-args JSON")?,
                    "prefill",
                )?,
                self.finalize_engine_args(
                    MockEngineArgs::from_json_str(decode)
                        .context("invalid --decode-engine-args JSON")?,
                    "decode",
                )?,
            )),
            (None, None) => {
                let mut prefill = self.engine_args()?;
                let mut decode = prefill.clone();
                prefill.worker_type = WorkerType::Prefill;
                decode.worker_type = WorkerType::Decode;
                Ok((prefill, decode))
            }
            _ => anyhow::bail!(
                "disaggregated offline mode requires both --prefill-engine-args and --decode-engine-args"
            ),
        }
    }

    fn replay_router_config(&self) -> Result<Option<ReplayKvRouterConfig>> {
        if self.router_config.is_none() && self.router_policy_config.is_none() {
            return Ok(None);
        }
        let mut value = match &self.router_config {
            Some(json) => serde_json::from_str::<serde_json::Value>(json)
                .context("invalid --router-config JSON")?,
            None => serde_json::json!({}),
        };
        let object = value
            .as_object_mut()
            .context("--router-config must contain a JSON object")?;
        if let Some(path) = &self.router_policy_config {
            object.insert(
                "router_policy_config".to_string(),
                serde_json::Value::String(path.display().to_string()),
            );
        }
        let config = serde_json::from_value::<ReplayKvRouterConfig>(value)
            .context("invalid Dynamo router configuration")?
            .with_policy_model_name(self.router_model_name.clone());
        config
            .validate_config()
            .map_err(anyhow::Error::msg)
            .context("invalid Dynamo router configuration")?;
        Ok(Some(config))
    }

    fn build_native(&self) -> Result<Box<dyn SteppableReplay>> {
        let router_config = self.replay_router_config()?;
        let mut engine: Box<dyn SteppableReplay> = match self.topology {
            // The routed runtime is the eventized form of one scheduler worker:
            // unlike the direct pass wrapper it can stop at an external clock
            // deadline, which rate/fixed/DAG co-simulation requires.
            OfflineTopology::Single => {
                let (args, estimator) = Self::configure_aic(self.engine_args()?)?;
                // Mirror `dynamo.replay`'s offline validation (lib/mocker
                // replay::validate): KV routing needs more than one worker, and
                // the single topology provisions exactly one.
                anyhow::ensure!(
                    self.router_mode != OfflineRouterMode::Kv,
                    "offline replay only supports router_mode=kv with more than one worker; topology=single provisions exactly one"
                );
                anyhow::ensure!(
                    args.dp_size == 1,
                    "offline replay only supports data_parallel_size=1, got {}",
                    args.dp_size
                );
                let estimator =
                    self.validate_prefill_load_estimator(router_config.as_ref(), estimator)?;
                if self.single_pass_engine && router_config.is_none() && estimator.is_none() {
                    // Dynamo's own offline single-runtime path: `execute_pass`
                    // prefills a mid-run arrival at its arrival instant, keeping
                    // saturated concurrency byte-exact with `dynamo.replay`.
                    Box::new(SteppableEngine::new(args))
                } else {
                    Box::new(SteppableAgg::new_with_router_config(
                        args,
                        1,
                        ReplayRouterMode::RoundRobin,
                        router_config,
                        estimator,
                    )?)
                }
            }
            OfflineTopology::Aggregated => {
                anyhow::ensure!(
                    self.workers > 0,
                    "offline aggregate workers must be positive"
                );
                let (args, estimator) = Self::configure_aic(self.engine_args()?)?;
                // Mirror `dynamo.replay`'s offline validation
                // (lib/mocker replay::validate_offline_router_mode /
                // validate_replay_args).
                if self.router_mode == OfflineRouterMode::Kv {
                    anyhow::ensure!(
                        self.workers > 1,
                        "offline replay only supports router_mode=kv when workers > 1, got {}",
                        self.workers
                    );
                }
                anyhow::ensure!(
                    args.dp_size == 1,
                    "offline replay only supports data_parallel_size=1, got {}",
                    args.dp_size
                );
                let estimator =
                    self.validate_prefill_load_estimator(router_config.as_ref(), estimator)?;
                if self.single_pass_engine
                    && self.workers == 1
                    && self.router_mode == OfflineRouterMode::RoundRobin
                    && router_config.is_none()
                    && estimator.is_none()
                {
                    Box::new(SteppableEngine::new(args))
                } else {
                    Box::new(SteppableAgg::new_with_router_config(
                        args,
                        self.workers,
                        self.router_mode.into(),
                        router_config,
                        estimator,
                    )?)
                }
            }
            OfflineTopology::Disaggregated => {
                anyhow::ensure!(
                    self.prefill_workers > 0,
                    "offline prefill workers must be positive"
                );
                anyhow::ensure!(
                    self.decode_workers > 0,
                    "offline decode workers must be positive"
                );
                let (prefill_args, decode_args) = self.disaggregated_engine_args()?;
                let (prefill_args, estimator) = Self::configure_aic(prefill_args)?;
                let (decode_args, _) = Self::configure_aic(decode_args)?;
                // Mirror `dynamo.replay`'s disaggregation validation
                // (lib/mocker replay::validate_disagg_args). The worker-type
                // guards are omitted: aiperf's topology selection is
                // authoritative and the runtime assigns prefill/decode roles.
                anyhow::ensure!(
                    prefill_args.engine_type != EngineType::Trtllm
                        && decode_args.engine_type != EngineType::Trtllm,
                    "disaggregation does not support TRT-LLM"
                );
                anyhow::ensure!(
                    prefill_args.engine_type == decode_args.engine_type,
                    "disaggregation requires matching prefill/decode engine_type, got {:?} and {:?}",
                    prefill_args.engine_type,
                    decode_args.engine_type
                );
                anyhow::ensure!(
                    prefill_args.dp_size == 1 && decode_args.dp_size == 1,
                    "disaggregation only supports data_parallel_size=1, got prefill={} decode={}",
                    prefill_args.dp_size,
                    decode_args.dp_size
                );
                anyhow::ensure!(
                    prefill_args.block_size == decode_args.block_size,
                    "disaggregation requires matching prefill/decode block_size, got {} and {}",
                    prefill_args.block_size,
                    decode_args.block_size
                );
                let estimator =
                    self.validate_prefill_load_estimator(router_config.as_ref(), estimator)?;
                Box::new(SteppableDisagg::new_with_router_config(
                    prefill_args,
                    decode_args,
                    self.prefill_workers,
                    self.decode_workers,
                    self.router_mode.into(),
                    router_config,
                    estimator,
                )?)
            }
        };
        engine.set_capture_per_request(self.capture_per_request);
        engine.set_sla_thresholds(self.sla);
        Ok(engine)
    }

    fn build(&self) -> Result<Box<dyn SteppableReplay>> {
        NativeDynamoEngineFactory.build(self)
    }
}

impl OfflineEngineFactory for NativeDynamoEngineFactory {
    fn build(&self, config: &OfflineEngineConfig) -> Result<Box<dyn SteppableReplay>> {
        config.build_native()
    }
}

const BACKEND_OWNED_SHARED_FIELDS: usize = 5;

/// Evidence that the two co-observed shared summaries serialized to exactly
/// the same bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OfflineMetricParity {
    /// Number of unique JSON fields in the common flat summary schema.
    pub shared_fields: usize,
    /// Request/event fields independently accumulated by both collectors.
    pub independently_accumulated_fields: usize,
    /// Engine-capacity fields imported from the backend that owns those facts.
    pub backend_owned_fields: usize,
    /// Size of either byte-identical compact JSON representation.
    pub serialized_bytes: usize,
}

/// Canonical byte representation for the metric schema shared by AIPerf's
/// compatibility collector and Dynamo's replay collector.
///
/// Future in-process backends can implement this trait for their native report
/// instead of weakening parity to a hand-picked field list.
pub trait CanonicalSharedMetrics {
    /// Serialize every field in the common summary schema without rounding,
    /// tolerance, field removal, or numeric normalization.
    fn canonical_shared_metric_bytes(&self) -> std::result::Result<Vec<u8>, serde_json::Error>;
}

impl CanonicalSharedMetrics for loadgen_core::collector::TraceSimulationReport {
    fn canonical_shared_metric_bytes(&self) -> std::result::Result<Vec<u8>, serde_json::Error> {
        serde_json::to_vec(self)
    }
}

impl CanonicalSharedMetrics for DynamoSimulationReport {
    fn canonical_shared_metric_bytes(&self) -> std::result::Result<Vec<u8>, serde_json::Error> {
        serde_json::to_vec(self)
    }
}

/// AIPerf's normal report plus Dynamo's native co-observed report.
pub struct OfflineRunReport {
    /// AIPerf compatibility and native-v2 metrics, built from live observer events.
    pub aiperf: OnlineRunReport,
    /// Dynamo-native scheduler/KV report from the same engine steps.
    pub dynamo: DynamoSimulationReport,
    /// Whole-summary byte-parity evidence produced before returning the run.
    pub parity: OfflineMetricParity,
}

/// Graph-mode AIPerf metrics plus Dynamo's native report from the same run.
pub struct OfflineGraphReport {
    /// Unified graph throughput/TTFT/native-metrics result.
    pub aiperf: GraphRpsReport,
    /// Full compatibility summary accumulated from the graph request stream.
    pub performance: loadgen_core::collector::TraceSimulationReport,
    /// Dynamo-native scheduler/KV report.
    pub dynamo: DynamoSimulationReport,
    /// Whole-summary byte-parity evidence produced before returning the run.
    pub parity: OfflineMetricParity,
}

/// Direct Graph-IR workload result over the same offline engine composition.
pub struct OfflineDirectGraphReport {
    /// Whole-root admission and trace completion outcomes.
    pub workload: GraphWorkloadReport,
    /// Full compatibility summary accumulated from node request events.
    pub performance: loadgen_core::collector::TraceSimulationReport,
    /// Native-v2 metric accumulator output from the same observations.
    pub native_metrics: AccumulatorSummary,
    /// Warmup-only native metrics when authored phases include warmup traffic.
    pub warmup_metrics: Option<AccumulatorSummary>,
    /// Final lifecycle snapshots in authored phase order.
    pub phases: Vec<PhaseStats>,
    /// Dynamo-native scheduler/KV report.
    pub dynamo: DynamoSimulationReport,
    /// Whole-summary byte-parity evidence produced before returning the run.
    pub parity: OfflineMetricParity,
}

/// One finalized offline graph request delivered to the orchestration layer.
///
/// The simulator owns request execution and measurement; the injected sink
/// decides whether to retain the already-complete record for adaptive control,
/// artifacts, or reporting. No model request or benchmark payload crosses this
/// seam.
pub struct OfflineGraphRequestRecord {
    /// Stable request identity assigned by the one run-scoped backend factory.
    pub uuid: Uuid,
    /// Unique graph execution-instance identity used for phase accounting.
    pub trace_id: String,
    /// Synthetic visible response emitted by the offline engine adapter.
    pub response_text: String,
    /// Complete transport-neutral metric input for this node request.
    pub ingest: RecordIngest,
}

/// Object-safe delivery seam for live offline graph request events.
pub trait OfflineGraphEventSink {
    /// Deliver the first output-token release for one node immediately.
    fn first_token(&self, uuid: Uuid, trace_id: &str) -> Result<()>;

    /// Deliver one terminal record before its root trace reports completion.
    fn record(&self, record: OfflineGraphRequestRecord) -> Result<()>;
}

struct DiscardOfflineGraphEvents;

impl OfflineGraphEventSink for DiscardOfflineGraphEvents {
    fn first_token(&self, _uuid: Uuid, _trace_id: &str) -> Result<()> {
        Ok(())
    }

    fn record(&self, _record: OfflineGraphRequestRecord) -> Result<()> {
        Ok(())
    }
}

/// Configuration for one phase-local backend over a shared offline engine.
pub struct OfflineGraphBackendConfig {
    /// Authoritative native-metrics phase tag.
    pub phase: MetricsPhase,
    /// Initial node-prefill capacity, when the phase controls it.
    pub prefill_concurrency: Option<usize>,
    /// Additional phase-local node policy such as deterministic cancellation.
    pub node_policy: Option<Rc<dyn NodeDispatchPolicy>>,
    /// Failure policy used by every trace executed in this phase.
    pub node_failure: Rc<dyn NodeFailurePolicy>,
    /// Live request-event consumer installed by the shared graph phase driver.
    pub events: Rc<dyn OfflineGraphEventSink>,
}

/// Run-scoped factory for phase-local graph execution backends.
///
/// Implementations share the simulator, clock, immutable segment store, and
/// request identity stream while returning a fresh cancellation latch and
/// prefill actuator for every authored phase.
pub trait OfflineGraphBackendFactory {
    /// Build one backend for an already validated phase.
    fn create_backend(&self, config: OfflineGraphBackendConfig) -> Result<Rc<dyn TracePlacement>>;
}

/// Backend-neutral result returned by an injected multi-phase graph driver.
pub struct OfflineGraphExecution {
    /// Aggregate root outcomes across all authored phases.
    pub workload: GraphWorkloadReport,
    /// Final lifecycle snapshots in authored phase order.
    pub phases: Vec<PhaseStats>,
}

/// Boxed local future used by deferred offline graph factories.
pub type DeferredOfflineGraphFuture = Pin<Box<dyn Future<Output = Result<OfflineGraphExecution>>>>;

/// Factory seam for graph orchestration over one initialized offline backend.
///
/// The implementation is expected to use the same graph phase driver as online
/// execution. Dynamo remains only the injected `{transport, clock}` backend.
pub trait DeferredOfflineGraphRunFactory {
    /// Construct the complete run after the shared clock and backend exist.
    fn create(
        self: Box<Self>,
        clock: Rc<dyn Clock>,
        backends: Rc<dyn OfflineGraphBackendFactory>,
    ) -> Result<DeferredOfflineGraphFuture>;
}

impl<F> DeferredOfflineGraphRunFactory for F
where
    F: FnOnce(
            Rc<dyn Clock>,
            Rc<dyn OfflineGraphBackendFactory>,
        ) -> Result<DeferredOfflineGraphFuture>
        + 'static,
{
    fn create(
        self: Box<Self>,
        clock: Rc<dyn Clock>,
        backends: Rc<dyn OfflineGraphBackendFactory>,
    ) -> Result<DeferredOfflineGraphFuture> {
        (*self)(clock, backends)
    }
}

/// Scheduled-workload AIPerf report plus Dynamo's native report.
pub struct OfflineScheduledReport {
    /// Authoritative profiling-phase result.
    pub aiperf: ScheduledRunReport,
    /// Run-wide compatibility report accumulated from the live observer stream
    /// across every warmup and profiling phase.
    pub performance: loadgen_core::collector::TraceSimulationReport,
    /// Final lifecycle snapshots in authored phase order.
    pub phases: Vec<crate::timing::PhaseStats>,
    /// Non-profiling phase reports retained without merging finalized summaries.
    pub auxiliary_phase_reports: Vec<ScheduledPhaseReport>,
    /// Dynamo-native scheduler/KV report.
    pub dynamo: DynamoSimulationReport,
    /// Whole-summary byte-parity evidence produced before returning the run.
    pub parity: OfflineMetricParity,
}

/// Typed output produced by one scheduled runtime factory before the backend
/// finalizes its engine-owned report.
///
/// `performance` is accumulated by one observer that spans the complete
/// engine/clock lifecycle. `profiling` remains the authoritative native report
/// source, while warmup reports stay independently inspectable rather than
/// being reconstructed through summary merging.
pub struct OfflineScheduledExecution {
    /// The one profiling-phase report.
    pub profiling: ScheduledRunReport,
    /// Whole-run compatibility metrics from the live cross-phase observer.
    pub performance: loadgen_core::collector::TraceSimulationReport,
    /// Final lifecycle snapshots in authored order.
    pub phases: Vec<crate::timing::PhaseStats>,
    /// Warmup or otherwise excluded phase reports.
    pub auxiliary_phase_reports: Vec<ScheduledPhaseReport>,
    aggregate_is_profiling: bool,
}

impl OfflineScheduledExecution {
    /// Adapt one legacy single-phase runtime without copying or merging any
    /// finalized metric summary.
    pub fn single(profiling: ScheduledRunReport) -> Self {
        Self {
            performance: profiling.performance.clone(),
            profiling,
            phases: Vec::new(),
            auxiliary_phase_reports: Vec::new(),
            aggregate_is_profiling: true,
        }
    }

    /// Select the unique profiling result from a shared phase-orchestrator run.
    pub fn phased(
        mut phased: PhasedScheduledRunReport,
        performance: loadgen_core::collector::TraceSimulationReport,
    ) -> Result<Self> {
        let profiling_indices = phased
            .reports
            .iter()
            .enumerate()
            .filter_map(|(index, report)| {
                (report.kind == crate::timing::PhaseKind::Profiling).then_some(index)
            })
            .collect::<Vec<_>>();
        anyhow::ensure!(
            profiling_indices.len() == 1,
            "offline scheduled execution requires exactly one profiling phase, got {}",
            profiling_indices.len()
        );
        let profiling = phased.reports.remove(profiling_indices[0]).report;
        let aggregate_is_profiling = phased.reports.is_empty();
        Ok(Self {
            profiling,
            performance,
            phases: phased.phases,
            auxiliary_phase_reports: phased.reports,
            aggregate_is_profiling,
        })
    }
}

fn finish_shared_metrics(
    aiperf: loadgen_core::collector::TraceSimulationReport,
    dynamo: DynamoSimulationReport,
) -> Result<(
    loadgen_core::collector::TraceSimulationReport,
    DynamoSimulationReport,
    OfflineMetricParity,
)> {
    // Deterministic (virtual-clock) runs require byte-exact AIPerf/Dynamo parity.
    finish_shared_metrics_enforcing(aiperf, dynamo, true)
}

/// Import backend-owned resource facts and, when `enforce_byte_parity`, prove the
/// AIPerf and Dynamo shared-metric serializations are byte-identical.
///
/// The wall-clock (`drive_real_with_source`) online mode passes `false`: request
/// and token counts still match, but latency/throughput are measured against real
/// timers and cannot be byte-identical to the engine's internal completion times,
/// so the proof is skipped while the same resource import and parity accounting
/// are retained.
fn finish_shared_metrics_enforcing(
    mut aiperf: loadgen_core::collector::TraceSimulationReport,
    dynamo: DynamoSimulationReport,
    enforce_byte_parity: bool,
) -> Result<(
    loadgen_core::collector::TraceSimulationReport,
    DynamoSimulationReport,
    OfflineMetricParity,
)> {
    let independently_accumulated =
        aiperf.request_counts.num_requests != 0 || dynamo.request_counts.num_requests == 0;
    if !independently_accumulated {
        // The native runner deliberately omits a second compatibility observer
        // for a single-phase Dynamo run. Dynamo already retains these exact
        // request/token facts for its canonical replay report; AIPerf continues
        // to own its richer native accumulator from the same callback stream.
        aiperf = compatibility_report_from_dynamo(&dynamo);
    }

    // Provisioned resources are properties of the injected engine, not request
    // observer events. Import exactly those backend-owned values; every request,
    // token, latency, throughput, and cache metric remains independently
    // accumulated by both collectors from the same event stream.
    aiperf.throughput.prefill_worker_seconds = dynamo.throughput.prefill_worker_seconds;
    aiperf.throughput.decode_worker_seconds = dynamo.throughput.decode_worker_seconds;
    aiperf.throughput.prefill_gpus_per_worker = dynamo.throughput.prefill_gpus_per_worker;
    aiperf.throughput.decode_gpus_per_worker = dynamo.throughput.decode_gpus_per_worker;
    aiperf.throughput.gpu_hours = dynamo.throughput.gpu_hours;
    // The backend collector owns the configured SLA thresholds and evaluates
    // them from the same request event stream. Copy its optional classification
    // before the byte-parity proof so the AIPerf compatibility report exposes
    // the canonical goodput values without recomputing from aggregate percentiles.
    aiperf.goodput =
        dynamo
            .goodput
            .as_ref()
            .map(|goodput| loadgen_core::collector::TraceGoodputStats {
                completed_requests: goodput.completed_requests,
                request_throughput_rps: goodput.request_throughput_rps,
                output_throughput_tok_s: goodput.output_throughput_tok_s,
            });

    let aiperf_bytes = aiperf
        .canonical_shared_metric_bytes()
        .context("failed to serialize AIPerf shared metrics")?;
    let dynamo_bytes = dynamo
        .canonical_shared_metric_bytes()
        .context("failed to serialize Dynamo shared metrics")?;
    if enforce_byte_parity && aiperf_bytes != dynamo_bytes {
        anyhow::bail!(
            "AIPerf/Dynamo shared metric byte mismatch: {}",
            shared_metric_diff(&aiperf_bytes, &dynamo_bytes)
        );
    }
    let shared_fields = serde_json::from_slice::<serde_json::Value>(&aiperf_bytes)?
        .as_object()
        .map_or(0, serde_json::Map::len);
    let backend_owned_fields = if independently_accumulated {
        BACKEND_OWNED_SHARED_FIELDS + if dynamo.goodput.is_some() { 3 } else { 0 }
    } else {
        shared_fields
    };
    let parity = OfflineMetricParity {
        shared_fields,
        independently_accumulated_fields: shared_fields.saturating_sub(backend_owned_fields),
        backend_owned_fields,
        serialized_bytes: aiperf_bytes.len(),
    };
    Ok((aiperf, dynamo, parity))
}

fn compatibility_report_from_dynamo(
    dynamo: &DynamoSimulationReport,
) -> loadgen_core::collector::TraceSimulationReport {
    use loadgen_core::collector::{
        TraceDistributionStats, TraceGoodputStats, TraceInterTokenLatencyStats, TraceLatencyStats,
        TraceRequestCounts, TraceSimulationReport, TraceThroughputStats,
    };

    fn distribution(source: &DynamoDistributionStats) -> TraceDistributionStats {
        TraceDistributionStats {
            mean_ms: source.mean_ms,
            min_ms: source.min_ms,
            max_ms: source.max_ms,
            median_ms: source.median_ms,
            p75_ms: source.p75_ms,
            p90_ms: source.p90_ms,
            p95_ms: source.p95_ms,
            p99_ms: source.p99_ms,
            std_ms: source.std_ms,
        }
    }

    TraceSimulationReport {
        request_counts: TraceRequestCounts {
            num_requests: dynamo.request_counts.num_requests,
            completed_requests: dynamo.request_counts.completed_requests,
            total_input_tokens: dynamo.request_counts.total_input_tokens,
            total_output_tokens: dynamo.request_counts.total_output_tokens,
        },
        throughput: TraceThroughputStats {
            duration_ms: dynamo.throughput.duration_ms,
            wall_time_ms: dynamo.throughput.wall_time_ms,
            request_throughput_rps: dynamo.throughput.request_throughput_rps,
            input_throughput_tok_s: dynamo.throughput.input_throughput_tok_s,
            output_throughput_tok_s: dynamo.throughput.output_throughput_tok_s,
            total_throughput_tok_s: dynamo.throughput.total_throughput_tok_s,
            prefill_worker_seconds: dynamo.throughput.prefill_worker_seconds,
            decode_worker_seconds: dynamo.throughput.decode_worker_seconds,
            prefill_gpus_per_worker: dynamo.throughput.prefill_gpus_per_worker,
            decode_gpus_per_worker: dynamo.throughput.decode_gpus_per_worker,
            gpu_hours: dynamo.throughput.gpu_hours,
        },
        prefix_cache_reused_ratio: dynamo.prefix_cache_reused_ratio,
        first_admission_prefix_cache_reused_ratio: dynamo.first_admission_prefix_cache_reused_ratio,
        latency: TraceLatencyStats {
            ttft: distribution(&dynamo.latency.ttft),
            ttst: distribution(&dynamo.latency.ttst),
            tpot: distribution(&dynamo.latency.tpot),
            itl: TraceInterTokenLatencyStats {
                distribution: distribution(&dynamo.latency.itl.distribution),
                max_ms: dynamo.latency.itl.max_ms,
            },
            e2e: distribution(&dynamo.latency.e2e),
            output_token_throughput_per_user: distribution(
                &dynamo.latency.output_token_throughput_per_user,
            ),
        },
        goodput: dynamo.goodput.as_ref().map(|goodput| TraceGoodputStats {
            completed_requests: goodput.completed_requests,
            request_throughput_rps: goodput.request_throughput_rps,
            output_throughput_tok_s: goodput.output_throughput_tok_s,
        }),
        per_request: Vec::new(),
    }
}

fn inject_dynamo_goodput(metrics: &mut AccumulatorSummary, report: &DynamoSimulationReport) {
    let Some(goodput) = &report.goodput else {
        return;
    };
    metrics.insert_finite(
        MetricTag::GoodRequestCount,
        goodput.completed_requests as f64,
    );
    metrics.insert_finite(MetricTag::Goodput, goodput.request_throughput_rps);
    let total = report.request_counts.num_requests;
    metrics.insert_finite(
        MetricTag::GoodRequestFraction,
        if total == 0 {
            0.0
        } else {
            goodput.completed_requests as f64 / total as f64
        },
    );
}

/// Write canonical Dynamo per-request records as one compact JSON object per
/// line, preserving backend timestamps, worker assignment, ITLs, and terminal
/// classification without reconstructing them from aggregate metrics.
pub fn write_dynamo_per_request_jsonl(
    report: &DynamoSimulationReport,
    path: impl AsRef<Path>,
) -> Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating per-request directory {}", parent.display()))?;
    }
    let file = std::fs::File::create(path)
        .with_context(|| format!("opening per-request JSONL {}", path.display()))?;
    let mut writer = BufWriter::new(file);
    for record in &report.per_request {
        serde_json::to_writer(&mut writer, record)
            .with_context(|| format!("serializing per-request JSONL {}", path.display()))?;
        writer.write_all(b"\n")?;
    }
    writer
        .flush()
        .with_context(|| format!("writing per-request JSONL {}", path.display()))?;
    Ok(())
}

/// Write Dynamo's canonical aggregate replay report, including optional
/// goodput fields configured through the AIPerf offline frontend.
pub fn write_dynamo_report_json(
    report: &DynamoSimulationReport,
    path: impl AsRef<Path>,
) -> Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating Dynamo report directory {}", parent.display()))?;
    }
    let value = serde_json::to_value(report).context("serializing Dynamo report")?;
    let object = value
        .as_object()
        .context("Dynamo report did not serialize as a JSON object")?;
    let sorted = object.iter().collect::<BTreeMap<_, _>>();
    let payload =
        serde_json::to_string_pretty(&sorted).context("serializing sorted Dynamo report")?;
    let mut payload = python_json_number_exponents(&payload);
    payload.push('\n');
    std::fs::write(path, payload)
        .with_context(|| format!("writing Dynamo report {}", path.display()))?;
    Ok(())
}

fn python_json_number_exponents(payload: &str) -> String {
    let bytes = payload.as_bytes();
    let mut normalized = String::with_capacity(payload.len());
    let mut in_string = false;
    let mut escaped = false;
    let mut index = 0;

    while index < bytes.len() {
        let byte = bytes[index];
        if in_string {
            normalized.push(char::from(byte));
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                in_string = false;
            }
            index += 1;
            continue;
        }
        if byte == b'"' {
            in_string = true;
            normalized.push('"');
            index += 1;
            continue;
        }
        if byte != b'e' && byte != b'E' {
            normalized.push(char::from(byte));
            index += 1;
            continue;
        }

        let mut exponent = index + 1;
        let sign = if exponent < bytes.len() && (bytes[exponent] == b'+' || bytes[exponent] == b'-')
        {
            let sign = bytes[exponent];
            exponent += 1;
            sign
        } else {
            b'+'
        };
        let digits_start = exponent;
        while exponent < bytes.len() && bytes[exponent].is_ascii_digit() {
            exponent += 1;
        }
        if exponent == digits_start {
            normalized.push(char::from(byte));
            index += 1;
            continue;
        }

        normalized.push('e');
        normalized.push(char::from(sign));
        if exponent - digits_start == 1 {
            normalized.push('0');
        }
        normalized.push_str(&payload[digits_start..exponent]);
        index = exponent;
    }

    normalized
}

fn shared_metric_diff(aiperf: &[u8], dynamo: &[u8]) -> String {
    let Ok(aiperf) = serde_json::from_slice::<serde_json::Value>(aiperf) else {
        return "AIPerf canonical bytes are not valid JSON".to_string();
    };
    let Ok(dynamo) = serde_json::from_slice::<serde_json::Value>(dynamo) else {
        return "Dynamo canonical bytes are not valid JSON".to_string();
    };
    let (Some(aiperf), Some(dynamo)) = (aiperf.as_object(), dynamo.as_object()) else {
        return "canonical reports are not JSON objects".to_string();
    };
    let mut keys = aiperf.keys().chain(dynamo.keys()).collect::<Vec<_>>();
    keys.sort_unstable();
    keys.dedup();
    let differences = keys
        .into_iter()
        .filter_map(|key| {
            let left = aiperf.get(key);
            let right = dynamo.get(key);
            (left != right).then(|| format!("{key}: aiperf={left:?}, dynamo={right:?}"))
        })
        .take(12)
        .collect::<Vec<_>>();
    if differences.is_empty() {
        "values match after JSON parsing but raw serialization differs".to_string()
    } else {
        differences.join("; ")
    }
}

#[derive(Debug, Clone, Copy)]
struct RoutedTerminalEvent {
    at_ns: i64,
    status: DynamoTerminalStatus,
    actual_output_length: Option<usize>,
    latencies_ms: Option<(f64, f64)>,
}

/// Request facts exposed to the offline event-delivery policy at submission.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OfflineEventDeliveryRequest {
    /// Prompt-token count submitted to Dynamo.
    pub input_tokens: usize,
    /// Maximum output-token count requested from Dynamo.
    pub max_output_tokens: usize,
}

/// Policy seam controlling whether nonterminal Dynamo events wake their request task.
///
/// Terminal events always wake. Returning `false` permits a deterministic offline
/// phase to buffer timestamped admission/token events and replay them in order at
/// terminal, avoiding Tokio scheduling work when no live policy consumes TTFT.
pub trait OfflineEventDeliveryPolicy {
    /// Whether this request requires task delivery before its terminal event.
    fn requires_incremental_delivery(&self, request: OfflineEventDeliveryRequest) -> bool;
}

/// Deliver every Dynamo event to the request task as soon as simulated time reaches it.
#[derive(Debug, Default)]
pub struct IncrementalOfflineEventDelivery;

impl OfflineEventDeliveryPolicy for IncrementalOfflineEventDelivery {
    fn requires_incremental_delivery(&self, _request: OfflineEventDeliveryRequest) -> bool {
        true
    }
}

/// Buffer nonterminal events and wake the request task exactly once at terminal.
#[derive(Debug, Default)]
pub struct TerminalOfflineEventDelivery;

impl OfflineEventDeliveryPolicy for TerminalOfflineEventDelivery {
    fn requires_incremental_delivery(&self, _request: OfflineEventDeliveryRequest) -> bool {
        false
    }
}

struct Waiter {
    output_token_times_ms: RefCell<Vec<f64>>,
    first_pending_output_token_ns: Cell<Option<i64>>,
    admission: Cell<Option<(f64, usize)>>,
    terminal: Cell<Option<RoutedTerminalEvent>>,
    admission_routed: Cell<bool>,
    incremental_delivery: bool,
    signal: LocalSignal,
}

impl Waiter {
    fn new(incremental_delivery: bool) -> Self {
        Self {
            output_token_times_ms: RefCell::new(Vec::new()),
            first_pending_output_token_ns: Cell::new(None),
            admission: Cell::new(None),
            terminal: Cell::new(None),
            admission_routed: Cell::new(false),
            incremental_delivery,
            signal: LocalSignal::new(),
        }
    }
}

/// Single-consumer wake signal for the worker-local offline executor.
///
/// `EngineHost` and every waiter future run on one `LocalSet`, so Tokio's
/// thread-safe atomic notification is unnecessary. The generation recheck
/// closes the register-versus-notify race without introducing another clock.
struct LocalSignal {
    generation: Cell<u64>,
    waker: RefCell<Option<Waker>>,
}

impl LocalSignal {
    const fn new() -> Self {
        Self {
            generation: Cell::new(0),
            waker: RefCell::new(None),
        }
    }

    fn notify(&self) {
        self.generation.set(self.generation.get().wrapping_add(1));
        let waker = self.waker.borrow_mut().take();
        if let Some(waker) = waker {
            waker.wake();
        }
    }

    fn notified(&self) -> LocalNotified<'_> {
        LocalNotified {
            signal: self,
            observed_generation: self.generation.get(),
        }
    }
}

struct LocalNotified<'a> {
    signal: &'a LocalSignal,
    observed_generation: u64,
}

impl Future for LocalNotified<'_> {
    type Output = ();

    fn poll(self: Pin<&mut Self>, context: &mut TaskContext<'_>) -> Poll<Self::Output> {
        if self.signal.generation.get() != self.observed_generation {
            return Poll::Ready(());
        }
        let mut waker = self.signal.waker.borrow_mut();
        if waker
            .as_ref()
            .is_none_or(|registered| !registered.will_wake(context.waker()))
        {
            *waker = Some(context.waker().clone());
        }
        drop(waker);
        if self.signal.generation.get() != self.observed_generation {
            self.signal.waker.borrow_mut().take();
            Poll::Ready(())
        } else {
            Poll::Pending
        }
    }
}

type Waiters = RefCell<FxHashMap<Uuid, Rc<Waiter>>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ScheduledCancellation {
    at_ns: i64,
    seq: u64,
    uuid: Uuid,
}

impl Ord for ScheduledCancellation {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .at_ns
            .cmp(&self.at_ns)
            .then_with(|| other.seq.cmp(&self.seq))
    }
}

impl PartialOrd for ScheduledCancellation {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Single-threaded host joining the passive engine, shared clock, and request
/// completion mailboxes. Engine steps never hold a waiter borrow while waking a
/// task, so resumed dispatch futures can immediately remove their mailbox.
struct EngineHost {
    clock: Rc<dyn Clock>,
    /// The virtual clock, present only under the DES (`drive_sim_with_source`)
    /// driver. It is used solely to bound an engine step so it never overshoots
    /// a pending arrival/gate — the deterministic-ordering guarantee. Under the
    /// wall-clock (`drive_real_with_source`) driver this is `None`: arrivals
    /// arrive in real time and the driver's producer wakeup re-evaluates the
    /// next engine event, so no virtual look-ahead exists (or is wanted).
    sim_clock: Option<Rc<SimClock>>,
    /// Notified whenever a request enters the engine, so the wall-clock
    /// (`drive_real_with_source`) driver re-evaluates the next engine event
    /// instead of waiting out a now-stale sleep. Inert under the DES driver,
    /// which re-checks the source every pump round.
    submit_wakeup: Rc<Notify>,
    engine: RefCell<Box<dyn SteppableReplay>>,
    event_delivery: Rc<dyn OfflineEventDeliveryPolicy>,
    waiters: Waiters,
    next_cancellation_seq: Cell<u64>,
    cancellations: RefCell<BinaryHeap<ScheduledCancellation>>,
    cancellation_by_uuid: RefCell<FxHashMap<Uuid, (i64, u64)>>,
}

impl EngineHost {
    fn new(clock: Rc<SimClock>, config: &OfflineEngineConfig) -> Result<Rc<Self>> {
        Self::new_with_factory_and_delivery(
            clock,
            config,
            &NativeDynamoEngineFactory,
            Rc::new(IncrementalOfflineEventDelivery),
        )
    }

    /// Default engine host over a wall clock for the in-process online driver.
    ///
    /// Mirrors [`EngineHost::new`] but binds the native engine to any `Clock`
    /// (typically `RealClock`) with no virtual look-ahead, so the engine steps
    /// only to driver-supplied real-time deadlines (`drive_real_with_source`).
    fn new_real(clock: Rc<dyn Clock>, config: &OfflineEngineConfig) -> Result<Rc<Self>> {
        Self::new_real_with_factory_and_delivery(
            clock,
            config,
            &NativeDynamoEngineFactory,
            Rc::new(IncrementalOfflineEventDelivery),
        )
    }

    fn new_with_delivery(
        clock: Rc<SimClock>,
        config: &OfflineEngineConfig,
        event_delivery: Rc<dyn OfflineEventDeliveryPolicy>,
    ) -> Result<Rc<Self>> {
        Self::new_with_factory_and_delivery(
            clock,
            config,
            &NativeDynamoEngineFactory,
            event_delivery,
        )
    }

    #[cfg(test)]
    fn new_with_factory(
        clock: Rc<SimClock>,
        config: &OfflineEngineConfig,
        factory: &dyn OfflineEngineFactory,
    ) -> Result<Rc<Self>> {
        Self::new_with_factory_and_delivery(
            clock,
            config,
            factory,
            Rc::new(IncrementalOfflineEventDelivery),
        )
    }

    fn new_with_factory_and_delivery(
        clock: Rc<SimClock>,
        config: &OfflineEngineConfig,
        factory: &dyn OfflineEngineFactory,
        event_delivery: Rc<dyn OfflineEventDeliveryPolicy>,
    ) -> Result<Rc<Self>> {
        let dyn_clock: Rc<dyn Clock> = clock.clone();
        Self::new_over_clock(dyn_clock, Some(clock), config, factory, event_delivery)
    }

    /// Build over a wall clock (or any `Clock`) with no virtual look-ahead.
    ///
    /// Used by the real-time (`drive_real_with_source`) driver: the engine steps
    /// only to the driver-supplied event time and never bounds itself by a
    /// virtual next-event, since arrivals occur in real time.
    fn new_real_with_factory_and_delivery(
        clock: Rc<dyn Clock>,
        config: &OfflineEngineConfig,
        factory: &dyn OfflineEngineFactory,
        event_delivery: Rc<dyn OfflineEventDeliveryPolicy>,
    ) -> Result<Rc<Self>> {
        Self::new_over_clock(clock, None, config, factory, event_delivery)
    }

    fn new_over_clock(
        clock: Rc<dyn Clock>,
        sim_clock: Option<Rc<SimClock>>,
        config: &OfflineEngineConfig,
        factory: &dyn OfflineEngineFactory,
        event_delivery: Rc<dyn OfflineEventDeliveryPolicy>,
    ) -> Result<Rc<Self>> {
        Ok(Rc::new(Self {
            clock,
            sim_clock,
            submit_wakeup: Rc::new(Notify::new()),
            engine: RefCell::new(factory.build(config)?),
            event_delivery,
            waiters: RefCell::new(FxHashMap::default()),
            next_cancellation_seq: Cell::new(0),
            cancellations: RefCell::new(BinaryHeap::new()),
            cancellation_by_uuid: RefCell::new(FxHashMap::default()),
        }))
    }

    /// Wakeup notified on each request submission, for the wall-clock driver.
    fn submit_wakeup(&self) -> Rc<Notify> {
        self.submit_wakeup.clone()
    }

    fn submit(&self, request: DirectRequest) -> Result<(Uuid, Rc<Waiter>)> {
        let delivery_request = OfflineEventDeliveryRequest {
            input_tokens: request.tokens.len(),
            max_output_tokens: request.max_output_tokens,
        };
        let incremental_delivery = self
            .event_delivery
            .requires_incremental_delivery(delivery_request);
        let mut engine = self.engine.borrow_mut();
        engine.advance_now_ms(ns_to_ms(self.clock.now_ns()));
        let uuid = engine.submit(request)?;
        drop(engine);

        let waiter = Rc::new(Waiter::new(incremental_delivery));
        if self
            .waiters
            .borrow_mut()
            .insert(uuid, waiter.clone())
            .is_some()
        {
            anyhow::bail!("duplicate in-flight Dynamo request id {uuid}");
        }
        // New work is queued: nudge the wall-clock driver to re-evaluate.
        self.submit_wakeup.notify_one();
        Ok((uuid, waiter))
    }

    fn release(&self, uuid: Uuid) {
        self.waiters.borrow_mut().remove(&uuid);
        self.cancellation_by_uuid.borrow_mut().remove(&uuid);
    }

    fn schedule_cancellation(&self, uuid: Uuid, delay_ns: i64) -> Result<()> {
        anyhow::ensure!(delay_ns >= 0, "cancellation delay must be non-negative");
        anyhow::ensure!(
            self.waiters.borrow().contains_key(&uuid),
            "cannot schedule cancellation for unknown request {uuid}"
        );
        let at_ns = self
            .clock
            .now_ns()
            .checked_add(delay_ns)
            .context("offline cancellation deadline overflow")?;
        let seq = self.next_cancellation_seq.get();
        self.next_cancellation_seq.set(seq.saturating_add(1));
        self.cancellation_by_uuid
            .borrow_mut()
            .insert(uuid, (at_ns, seq));
        self.cancellations
            .borrow_mut()
            .push(ScheduledCancellation { at_ns, seq, uuid });
        Ok(())
    }

    fn next_cancellation_ns(&self) -> Option<i64> {
        loop {
            let next = self.cancellations.borrow().peek().copied()?;
            if self
                .cancellation_by_uuid
                .borrow()
                .get(&next.uuid)
                .is_some_and(|active| *active == (next.at_ns, next.seq))
            {
                return Some(next.at_ns.max(self.clock.now_ns()));
            }
            self.cancellations.borrow_mut().pop();
        }
    }

    fn pop_due_cancellations(&self, now_ns: i64) -> Vec<Uuid> {
        let mut due = Vec::new();
        loop {
            let Some(next) = self.cancellations.borrow().peek().copied() else {
                break;
            };
            if next.at_ns > now_ns {
                break;
            }
            self.cancellations.borrow_mut().pop();
            let active = self.cancellation_by_uuid.borrow().get(&next.uuid).copied();
            if active == Some((next.at_ns, next.seq)) {
                self.cancellation_by_uuid.borrow_mut().remove(&next.uuid);
                due.push(next.uuid);
            }
        }
        due
    }

    fn cancel_due(&self, now_ns: i64) -> Result<bool> {
        let due = self.pop_due_cancellations(now_ns);
        if due.is_empty() {
            return Ok(false);
        }
        let now_ms = ns_to_ms(now_ns);
        let mut events = Vec::with_capacity(due.len());
        {
            let mut engine = self.engine.borrow_mut();
            engine.advance_now_ms(now_ms);
            for uuid in due {
                let event = engine.cancel(uuid)?.with_context(|| {
                    format!("Dynamo request {uuid} became terminal before its cancellation event")
                })?;
                events.push(event);
            }
        }
        self.route(now_ms, now_ns, events);
        Ok(true)
    }

    #[cfg(test)]
    fn take_report(&self) -> DynamoSimulationReport {
        let wall_ms = ns_to_ms(self.clock.now_ns());
        self.take_report_at(wall_ms)
    }

    fn take_report_at(&self, wall_ms: f64) -> DynamoSimulationReport {
        self.engine.borrow_mut().take_report(wall_ms)
    }

    fn route(&self, at_ms: f64, at_ns: i64, events: Vec<EngineEvent>) {
        for event in events {
            let waiter = self.waiters.borrow().get(&event.uuid).cloned();
            let needs_admission = waiter
                .as_ref()
                .is_some_and(|waiter| !waiter.admission_routed.get());
            let (admission, actual_output_length, latencies_ms) = {
                let engine = self.engine.borrow();
                let admission = needs_admission
                    .then(|| engine.request_admission(event.uuid))
                    .flatten();
                if event.terminal_status.is_some() {
                    (
                        admission,
                        engine.actual_output_length(event.uuid),
                        engine.request_latencies(event.uuid),
                    )
                } else {
                    (admission, None, None)
                }
            };
            if event.terminal_status.is_some() {
                self.cancellation_by_uuid.borrow_mut().remove(&event.uuid);
            }
            if let Some(waiter) = waiter {
                let terminal = event.terminal_status.is_some();
                if admission.is_some() {
                    waiter.admission_routed.set(true);
                    waiter.admission.set(admission);
                }
                if event.emitted_token {
                    let mut output_token_times_ms = waiter.output_token_times_ms.borrow_mut();
                    if output_token_times_ms.is_empty() {
                        waiter.first_pending_output_token_ns.set(Some(at_ns));
                    }
                    output_token_times_ms.push(at_ms);
                }
                if let Some(status) = event.terminal_status {
                    waiter.terminal.set(Some(RoutedTerminalEvent {
                        at_ns,
                        status,
                        actual_output_length,
                        latencies_ms,
                    }));
                }
                if terminal || waiter.incremental_delivery {
                    waiter.signal.notify();
                }
            }
        }
    }
}

impl SimEventSource for EngineHost {
    fn next_event_ns(&self) -> std::result::Result<Option<i64>, SimDriveError> {
        let next_engine_ns = {
            let mut engine = self.engine.borrow_mut();
            engine.next_event_ms().or_else(|| {
                // Routed runtimes can have newly submitted current-timestamp
                // work before it has been lowered into their event heap.
                (!engine.is_idle()).then(|| engine.now_ms())
            })
        }
        .map(ms_to_ns)
        .transpose()?;
        Ok(match (next_engine_ns, self.next_cancellation_ns()) {
            (Some(engine), Some(cancel)) => Some(engine.min(cancel)),
            (Some(engine), None) => Some(engine),
            (None, Some(cancel)) => Some(cancel),
            (None, None) => None,
        })
    }

    fn set_time_ns(&self, now_ns: i64) -> std::result::Result<(), SimDriveError> {
        self.engine.borrow_mut().advance_now_ms(ns_to_ms(now_ns));
        Ok(())
    }

    fn step(&self, now_ns: i64) -> std::result::Result<SimStep, SimDriveError> {
        if self
            .cancel_due(now_ns)
            .map_err(|error| SimDriveError::EventSource(error.to_string()))?
        {
            return Ok(SimStep {
                end_ns: now_ns,
                made_progress: true,
            });
        }
        let (before_ms, before_in_flight, before_event_ms) = {
            let mut engine = self.engine.borrow_mut();
            engine.advance_now_ms(ns_to_ms(now_ns));
            let before_ms = engine.now_ms();
            let before_in_flight = engine.in_flight();
            let before_event_ms = engine.next_event_ms();
            (before_ms, before_in_flight, before_event_ms)
        };

        let clock_event = self
            .sim_clock
            .as_ref()
            .and_then(|sim_clock| sim_clock.next_event_time());
        let until_ns = match (clock_event, self.next_cancellation_ns()) {
            (Some(clock), Some(cancel)) => Some(clock.min(cancel)),
            (Some(clock), None) => Some(clock),
            (None, Some(cancel)) => Some(cancel),
            (None, None) => None,
        };
        let until_ms = until_ns.map(ns_to_ms).unwrap_or(f64::INFINITY);
        let outcome = self
            .engine
            .borrow_mut()
            .step_until(until_ms)
            .map_err(|error| SimDriveError::EventSource(error.to_string()))?;
        let end_ns = ms_to_ns(outcome.end_ms)?;
        let had_events = !outcome.events.is_empty();
        self.route(outcome.end_ms, end_ns, outcome.events);

        let (after_in_flight, after_event_ms) = {
            let mut engine = self.engine.borrow_mut();
            (engine.in_flight(), engine.next_event_ms())
        };
        Ok(SimStep {
            end_ns,
            made_progress: outcome.end_ms > before_ms
                || had_events
                || after_in_flight != before_in_flight
                || after_event_ms != before_event_ms,
        })
    }

    fn is_idle(&self) -> bool {
        self.engine.borrow().is_idle()
    }
}

fn ms_to_ns(ms: f64) -> std::result::Result<i64, SimDriveError> {
    if !ms.is_finite() || ms < 0.0 || ms >= i64::MAX as f64 / NS_PER_MS {
        return Err(SimDriveError::EventSource(format!(
            "Dynamo returned invalid simulation time {ms}ms"
        )));
    }
    Ok((ms * NS_PER_MS).round() as i64)
}

#[inline]
fn ns_to_ms(ns: i64) -> f64 {
    ns as f64 / NS_PER_MS
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// Request-to-token seam for the mock engine. Endpoint-specific offline
/// adapters can implement this trait without changing the engine host.
pub trait OfflineRequestEncoder<R> {
    /// Convert one endpoint request into the token ids consumed by the engine.
    fn encode(&self, request: &R) -> Result<Vec<u32>>;
}

/// Source-trace hash conversion seam for offline simulator requests.
///
/// The unified dataset retains backend-neutral authored block identities. A
/// simulator adapter implements this trait to lower those identities into its
/// request representation without inferring them from materialized text.
pub trait OfflineTraceHashEncoder {
    /// Convert one authored hash sequence into the exact prompt tokens consumed
    /// by the simulator.
    fn encode(&self, hash_ids: &[i64], block_size: usize, input_length: usize) -> Result<Vec<u32>>;
}

/// Dynamo-native source-trace hash conversion.
///
/// This delegates to Dynamo's public [`TurnTrace::synthesize_tokens`] method,
/// which is the same conversion used by official Mooncake replay before the
/// engine derives router block hashes.
#[derive(Debug, Clone, Copy, Default)]
pub struct DynamoTraceHashEncoder;

impl OfflineTraceHashEncoder for DynamoTraceHashEncoder {
    fn encode(&self, hash_ids: &[i64], block_size: usize, input_length: usize) -> Result<Vec<u32>> {
        let hash_ids = hash_ids
            .iter()
            .copied()
            .map(|hash_id| {
                u64::try_from(hash_id).with_context(|| {
                    format!("trace hash_id {hash_id} cannot be represented as u64")
                })
            })
            .collect::<Result<Vec<_>>>()?;
        TurnTrace {
            input_length,
            hash_ids,
            ..TurnTrace::default()
        }
        .synthesize_tokens(block_size)
        .context("converting authored hash_ids through Dynamo's trace prompt representation")
    }
}

/// Graph-message tokenization seam for an offline engine adapter.
pub trait OfflineGraphRequestEncoder {
    /// Convert already-materialized graph message wires to an exact input size.
    fn encode_graph_messages(&self, wires: &[Bytes], requested: usize) -> Result<Vec<u32>>;

    /// Encode authored graph wires without an externally supplied synthetic
    /// length. Direct Graph-IR adapters use this path; benchmark fixtures may
    /// continue to request an exact synthetic input size above.
    fn encode_graph_messages_exact(&self, wires: &[Bytes]) -> Result<Vec<u32>> {
        self.encode_graph_messages(wires, wires.len().max(1))
    }
}

/// Tiktoken-backed encoder for OpenAI-compatible request and graph wires.
pub struct OpenAiRequestEncoder {
    tokenizer: TiktokenTokenizer,
}

impl Default for OpenAiRequestEncoder {
    fn default() -> Self {
        Self {
            tokenizer: TiktokenTokenizer::builtin(),
        }
    }
}

impl OpenAiRequestEncoder {
    fn encode_message_values(&self, messages: &[serde_json::Value]) -> Result<Vec<u32>> {
        let mut tokens = Vec::new();
        for message in messages {
            let wire = serde_json::to_string(message)?;
            tokens.extend(self.tokenizer.encode(&wire)?);
        }
        Ok(tokens)
    }

    fn encode_json(&self, value: &serde_json::Value) -> Result<Vec<u32>> {
        if let Some(messages) = value.get("messages").and_then(serde_json::Value::as_array) {
            self.encode_message_values(messages)
        } else {
            Ok(self.tokenizer.encode(&serde_json::to_string(value)?)?)
        }
    }

    fn fit_synthetic_prompt(&self, text: &str, requested: usize) -> Result<Vec<u32>> {
        let tokens = self.tokenizer.encode(text)?;
        Ok(Self::fit_encoded_tokens(
            tokens,
            requested,
            fnv1a64(text.as_bytes()),
        ))
    }

    fn fit_encoded_tokens(mut tokens: Vec<u32>, requested: usize, seed: u64) -> Vec<u32> {
        tokens.truncate(requested);
        while tokens.len() < requested {
            let index = tokens.len() as u64;
            tokens.push(seed.wrapping_add(index.wrapping_mul(0x9e37_79b9)) as u32);
        }
        tokens
    }

    fn encode_wires(&self, wires: &[Bytes], requested: usize) -> Result<Vec<u32>> {
        let mut values = Vec::with_capacity(wires.len());
        for wire in wires {
            values.push(serde_json::from_slice(wire).with_context(|| {
                format!(
                    "graph message is not valid JSON: {}",
                    String::from_utf8_lossy(wire)
                )
            })?);
        }
        let tokens = self.encode_message_values(&values)?;
        let seed = wires.iter().fold(0xcbf2_9ce4_8422_2325_u64, |hash, wire| {
            fnv1a64(wire).wrapping_mul(hash)
        });
        Ok(Self::fit_encoded_tokens(tokens, requested, seed))
    }
}

impl OfflineRequestEncoder<Request> for OpenAiRequestEncoder {
    fn encode(&self, request: &Request) -> Result<Vec<u32>> {
        if let Some(bytes) = &request.request_body_bytes {
            let tokens = match serde_json::from_slice(bytes) {
                Ok(value) => self.encode_json(&value),
                Err(_) => Ok(self.tokenizer.encode(&String::from_utf8_lossy(bytes))?),
            }?;
            return Ok(Self::fit_encoded_tokens(
                tokens,
                request.input_length,
                fnv1a64(bytes),
            ));
        }
        if let Some(value) = &request.request_body {
            let wire = serde_json::to_vec(value)?;
            return Ok(Self::fit_encoded_tokens(
                self.encode_json(value)?,
                request.input_length,
                fnv1a64(&wire),
            ));
        }
        self.fit_synthetic_prompt(
            request.prompt_text.as_deref().unwrap_or_default(),
            request.input_length,
        )
    }
}

impl OfflineGraphRequestEncoder for OpenAiRequestEncoder {
    fn encode_graph_messages(&self, wires: &[Bytes], requested: usize) -> Result<Vec<u32>> {
        self.encode_wires(wires, requested)
    }

    fn encode_graph_messages_exact(&self, wires: &[Bytes]) -> Result<Vec<u32>> {
        let mut values = Vec::with_capacity(wires.len());
        for wire in wires {
            values.push(serde_json::from_slice(wire).with_context(|| {
                format!(
                    "graph message is not valid JSON: {}",
                    String::from_utf8_lossy(wire)
                )
            })?);
        }
        self.encode_message_values(&values)
    }
}

type CachedTracePrompt = (Handle, usize, Box<[u32]>);

/// `RequestSink<Request>` and scheduled-turn adapter over one [`EngineHost`].
struct DynosimSink {
    host: Rc<EngineHost>,
    clock: Rc<dyn Clock>,
    start_ns: i64,
    model: String,
    request_encoder: Rc<dyn OfflineRequestEncoder<Request>>,
    graph_encoder: Rc<dyn OfflineGraphRequestEncoder>,
    trace_hash_encoder: Rc<dyn OfflineTraceHashEncoder>,
    last_trace_prompt: RefCell<Option<CachedTracePrompt>>,
}

impl DynosimSink {
    fn new(host: Rc<EngineHost>, clock: Rc<dyn Clock>, model: String) -> Rc<Self> {
        let encoder = Rc::new(OpenAiRequestEncoder::default());
        Self::new_with_encoders(host, clock, model, encoder.clone(), encoder)
    }

    fn new_with_encoders(
        host: Rc<EngineHost>,
        clock: Rc<dyn Clock>,
        model: String,
        request_encoder: Rc<dyn OfflineRequestEncoder<Request>>,
        graph_encoder: Rc<dyn OfflineGraphRequestEncoder>,
    ) -> Rc<Self> {
        Self::new_with_all_encoders(
            host,
            clock,
            model,
            request_encoder,
            graph_encoder,
            Rc::new(DynamoTraceHashEncoder),
        )
    }

    fn new_with_all_encoders(
        host: Rc<EngineHost>,
        clock: Rc<dyn Clock>,
        model: String,
        request_encoder: Rc<dyn OfflineRequestEncoder<Request>>,
        graph_encoder: Rc<dyn OfflineGraphRequestEncoder>,
        trace_hash_encoder: Rc<dyn OfflineTraceHashEncoder>,
    ) -> Rc<Self> {
        Rc::new(Self {
            host,
            start_ns: clock.now_ns(),
            clock,
            model,
            request_encoder,
            graph_encoder,
            trace_hash_encoder,
            last_trace_prompt: RefCell::new(None),
        })
    }

    fn encode_trace_prompt(
        &self,
        stored_hash_ids: &crate::multiturn::StoredTraceHashIds,
        input_length: usize,
    ) -> Result<Vec<u32>> {
        let handle = stored_hash_ids.handle();
        // Trace segments are content-addressed, so adjacent repeated rows share
        // a handle. One bounded prototype avoids rebuilding source hashes on the
        // LocalSet without turning the segment pool into an unbounded token cache.
        if let Some((cached_handle, cached_length, tokens)) =
            self.last_trace_prompt.borrow().as_ref()
            && *cached_handle == handle
            && *cached_length == input_length
        {
            return Ok(tokens.to_vec());
        }
        let (hash_ids, block_size) = stored_hash_ids.resolve()?;
        let tokens = self
            .trace_hash_encoder
            .encode(hash_ids, block_size, input_length)?;
        *self.last_trace_prompt.borrow_mut() =
            Some((handle, input_length, tokens.clone().into_boxed_slice()));
        Ok(tokens)
    }

    fn relative_ms(&self, at_ns: i64) -> f64 {
        at_ns.saturating_sub(self.start_ns) as f64 / NS_PER_MS
    }

    async fn dispatch_tokens(
        &self,
        uuid: Uuid,
        mut tokens: Vec<u32>,
        max_output_tokens: usize,
        cancel_after_ns: Option<i64>,
        observer: &dyn RequestObserver,
        on_first_token: &dyn Fn(i64),
    ) -> Result<HttpDispatchResult> {
        if tokens.is_empty() {
            tokens.push(0);
        }
        let start_ns = self.clock.now_ns();
        let prompt_tokens = tokens.len();
        let request = DirectRequest {
            tokens,
            max_output_tokens,
            output_token_ids: None,
            uuid: Some(uuid),
            dp_rank: 0,
            arrival_timestamp_ms: Some(ns_to_ms(start_ns)),
            priority: 0,
            strict_priority: 0,
            policy_class: None,
        };
        let (assigned_uuid, waiter) = self.host.submit(request)?;
        if let Some(delay_ns) = cancel_after_ns {
            self.host.schedule_cancellation(assigned_uuid, delay_ns)?;
        }
        let mut first_token = false;
        let mut admitted = false;
        let mut observed_output_tokens = 0_usize;
        let mut ready_output_token_times_ms = Vec::new();
        let observer_origin_ms = ns_to_ms(self.start_ns);

        // Coalesced delivery cannot expose an admission or token before the
        // terminal wake, so probing the empty mailbox first is pure overhead.
        if !waiter.incremental_delivery {
            waiter.signal.notified().await;
        }

        loop {
            {
                let mut pending = waiter.output_token_times_ms.borrow_mut();
                std::mem::swap(&mut ready_output_token_times_ms, &mut *pending);
            }
            let first_pending_output_token_ns = waiter.first_pending_output_token_ns.take();
            let terminal = waiter.terminal.take();
            if !admitted && let Some((admit_ms, reused_input_tokens)) = waiter.admission.take() {
                observer.on_admit(uuid, admit_ms - observer_origin_ms, reused_input_tokens);
                admitted = true;
            }
            if !ready_output_token_times_ms.is_empty() {
                observed_output_tokens += ready_output_token_times_ms.len();
                for timestamp in &mut ready_output_token_times_ms {
                    *timestamp -= observer_origin_ms;
                }
                observer.on_output_tokens(uuid, &ready_output_token_times_ms);
                if !first_token {
                    first_token = true;
                    on_first_token(
                        first_pending_output_token_ns
                            .expect("buffered Dynamo tokens retain their first timestamp")
                            .saturating_sub(start_ns),
                    );
                }
                ready_output_token_times_ms.clear();
            }
            if let Some(routed) = terminal {
                let terminal = map_terminal_status(routed.status);
                if terminal == ReplayTerminalStatus::Completed
                    && !first_token
                    && let Some((ttft_ms, _)) = routed.latencies_ms
                {
                    let ttft_ns = ms_to_ns(ttft_ms).map_err(anyhow::Error::msg)?;
                    observer.on_classified_token(
                        uuid,
                        ns_to_ms(start_ns) + ttft_ms - observer_origin_ms,
                        ObservedTokenKind::Output,
                    );
                    on_first_token(ttft_ns);
                }
                let completion_tokens = routed
                    .actual_output_length
                    .unwrap_or(observed_output_tokens);
                observer.on_usage(
                    uuid,
                    ObservedUsage {
                        prompt_tokens: Some(prompt_tokens),
                        completion_tokens: Some(completion_tokens),
                        ..ObservedUsage::default()
                    },
                );
                observer.on_terminal(uuid, terminal);
                self.host.release(assigned_uuid);
                return Ok(HttpDispatchResult {
                    start_ns,
                    end_ns: routed.at_ns,
                    status: None,
                    terminal,
                    response_text: if terminal == ReplayTerminalStatus::Completed {
                        synthetic_reply(uuid, completion_tokens)
                    } else {
                        String::new()
                    },
                    model_response: ModelResponseMetadata::default(),
                    prompt_tokens: u32::try_from(prompt_tokens).ok(),
                    completion_tokens: u32::try_from(completion_tokens).ok(),
                    http: RequestTrace::default(),
                });
            }
            waiter.signal.notified().await;
        }
    }
}

fn map_terminal_status(status: DynamoTerminalStatus) -> ReplayTerminalStatus {
    match status {
        DynamoTerminalStatus::Completed => ReplayTerminalStatus::Completed,
        DynamoTerminalStatus::Rejected => ReplayTerminalStatus::Rejected,
        DynamoTerminalStatus::Canceled => ReplayTerminalStatus::Canceled,
        DynamoTerminalStatus::Failed => ReplayTerminalStatus::Failed,
    }
}

fn synthetic_reply(uuid: Uuid, output_tokens: usize) -> String {
    let unit = format!("sim-{:08x} ", uuid.as_u128() as u32);
    let target_bytes = output_tokens.max(1).saturating_mul(4);
    let mut reply = String::with_capacity(target_bytes + unit.len());
    while reply.len() < target_bytes {
        reply.push_str(&unit);
    }
    reply
}

#[async_trait(?Send)]
impl RequestSink<Request> for DynosimSink {
    async fn dispatch(&self, request: Request, observer: &dyn RequestObserver) -> Result<()> {
        self.dispatch_collect(&request, observer, &|_| {}).await?;
        Ok(())
    }
}

impl DynosimSink {
    async fn dispatch_collect(
        &self,
        request: &Request,
        observer: &dyn RequestObserver,
        on_first_token: &dyn Fn(i64),
    ) -> Result<HttpDispatchResult> {
        let tokens = self.request_encoder.encode(request)?;
        self.dispatch_tokens(
            request.uuid,
            tokens,
            request.max_output_tokens,
            request.cancel_after_ns,
            observer,
            on_first_token,
        )
        .await
    }
}

#[async_trait(?Send)]
impl HttpRequestDispatcher for DynosimSink {
    fn inference_dimensions(&self, _request: &Request) -> InferenceDimensions {
        InferenceDimensions {
            endpoint_url: Some("dynosim://offline".to_string()),
            model: Some(self.model.clone()),
        }
    }

    async fn dispatch_collect(
        &self,
        request: Request,
        observer: &dyn RequestObserver,
        on_first_token: &dyn Fn(i64),
    ) -> Result<HttpDispatchResult> {
        self.dispatch_collect(&request, observer, on_first_token)
            .await
    }
}

#[async_trait(?Send)]
impl TurnDispatcher for DynosimSink {
    fn inference_dimensions(&self, turn: &TurnToSend) -> InferenceDimensions {
        InferenceDimensions {
            endpoint_url: Some("dynosim://offline".to_string()),
            model: turn
                .effective_model
                .clone()
                .or_else(|| Some(self.model.clone())),
        }
    }

    async fn dispatch_turn(
        &self,
        turn: TurnToSend,
        observer: &dyn RequestObserver,
        on_first_token: &dyn Fn(i64),
    ) -> Result<TurnDispatchOutcome> {
        let is_final_turn = turn.is_final_turn();
        let result = if let Some(stored_token_ids) = &turn.raw_token_ids {
            self.dispatch_tokens(
                turn.uuid,
                stored_token_ids.resolve()?.to_vec(),
                turn.max_output_tokens,
                turn.cancel_after_ns,
                observer,
                on_first_token,
            )
            .await?
        } else if let Some(stored_hash_ids) = &turn.trace_hash_ids {
            let tokens = self.encode_trace_prompt(stored_hash_ids, turn.input_length)?;
            self.dispatch_tokens(
                turn.uuid,
                tokens,
                turn.max_output_tokens,
                turn.cancel_after_ns,
                observer,
                on_first_token,
            )
            .await?
        } else {
            let request_body = if turn.request_body.is_none() {
                let messages = turn
                    .messages
                    .iter()
                    .map(|message| (message.role.as_str(), message.content.as_str()))
                    .collect::<Vec<_>>();
                Some(chat_request_body(
                    &self.model,
                    &messages,
                    turn.max_output_tokens,
                ))
            } else {
                None
            };
            self.dispatch_collect(
                &Request {
                    uuid: turn.uuid,
                    input_length: turn.input_length,
                    max_output_tokens: turn.max_output_tokens,
                    prompt_text: None,
                    request_body,
                    request_body_bytes: turn.request_body,
                    headers: turn.request_headers,
                    parameters: turn.request_parameters,
                    endpoint_path: turn.endpoint_path,
                    streaming: turn.streaming,
                    x_correlation_id: Some(turn.request_correlation_id),
                    is_final_turn,
                    cancel_after_ns: turn.cancel_after_ns,
                    url_index: turn.url_index,
                },
                observer,
                on_first_token,
            )
            .await?
        };
        Ok(TurnDispatchOutcome {
            start_ns: result.start_ns,
            end_ns: result.end_ns,
            terminal: result.terminal,
            response_text: result.response_text,
            model_response: result.model_response,
            prompt_tokens: result.prompt_tokens.map(u64::from),
            completion_tokens: result.completion_tokens.map(u64::from),
            http: result.http,
        })
    }
}

/// Run the ordinary concurrency/request-rate issuer against Dynamo in-process.
/// The same `run_paced_with_backend` future used online is polled by the shared
/// sleeper-plus-engine DES pump.
#[allow(clippy::too_many_arguments)]
pub fn run_paced_offline(
    engine_config: OfflineEngineConfig,
    model: String,
    workload: SkeletonWorkload,
    pattern: ArrivalPattern,
    rate: Option<f64>,
    smoothness: Option<f64>,
    concurrency: Option<usize>,
    prefill_concurrency: Option<usize>,
    stop: StopConfig,
    seed: u64,
    adaptive: Option<AdaptiveRunConfig>,
    ancillary: AncillaryTimingConfig,
) -> Result<OfflineRunReport> {
    ancillary.validate()?;
    let clock = Rc::new(SimClock::new());
    let host = EngineHost::new(clock.clone(), &engine_config)?;
    let sink: Rc<dyn HttpRequestDispatcher> = DynosimSink::new(host.clone(), clock.clone(), model);
    let result = Rc::new(RefCell::new(None));
    let result_for_body = result.clone();
    let clock_for_body: Rc<dyn Clock> = clock.clone();
    let start_ns = clock.now_ns();
    let source: Rc<dyn SimEventSource> = host.clone();
    let outcome = drive_sim_with_source(clock.clone(), source, move |_handle| async move {
        let run = run_paced_with_backend(
            clock_for_body,
            start_ns,
            sink,
            vec!["dynosim".to_string()],
            workload,
            pattern,
            rate,
            smoothness,
            concurrency,
            prefill_concurrency,
            stop,
            seed,
            adaptive,
            ancillary,
        )
        .await;
        *result_for_body.borrow_mut() = Some(run);
    })?;
    anyhow::ensure!(!outcome.deadlocked, "Dynamo offline run deadlocked");
    let mut aiperf = result
        .borrow_mut()
        .take()
        .context("Dynamo offline driver exited without a paced result")??;
    let dynamo = host.take_report_at(aiperf.performance.throughput.wall_time_ms);
    let (performance, dynamo, parity) = finish_shared_metrics(aiperf.performance, dynamo)?;
    aiperf.performance = performance;
    inject_dynamo_goodput(&mut aiperf.metrics, &dynamo);
    Ok(OfflineRunReport {
        aiperf,
        dynamo,
        parity,
    })
}

/// Run the ordinary concurrency/request-rate issuer against the in-process
/// Dynamo engine under the **wall clock** — the real-time in-process mode.
///
/// Identical to [`run_paced_offline`] (same sink, `run_paced_with_backend`
/// issuer, materializer, observer, and report) except the engine is stepped in
/// real time by [`drive_real_with_source`] instead of fast-forwarded by the DES
/// pump, and the AIPerf/Dynamo byte-exact parity proof is relaxed (real-timer
/// latencies cannot match the engine's internal completion times). Request and
/// token counts remain exact. Trace timing must already be speed-scaled or the
/// run takes the trace's real duration.
#[allow(clippy::too_many_arguments)]
pub fn run_paced_online(
    engine_config: OfflineEngineConfig,
    model: String,
    workload: SkeletonWorkload,
    pattern: ArrivalPattern,
    rate: Option<f64>,
    smoothness: Option<f64>,
    concurrency: Option<usize>,
    prefill_concurrency: Option<usize>,
    stop: StopConfig,
    seed: u64,
    adaptive: Option<AdaptiveRunConfig>,
    ancillary: AncillaryTimingConfig,
) -> Result<OfflineRunReport> {
    ancillary.validate()?;
    let clock: Rc<dyn Clock> = crate::clock::real_clock::RealClock::new();
    let host = EngineHost::new_real_with_factory_and_delivery(
        clock.clone(),
        &engine_config,
        &NativeDynamoEngineFactory,
        Rc::new(IncrementalOfflineEventDelivery),
    )?;
    let sink: Rc<dyn HttpRequestDispatcher> = DynosimSink::new(host.clone(), clock.clone(), model);
    let result = Rc::new(RefCell::new(None));
    let result_for_body = result.clone();
    let clock_for_body: Rc<dyn Clock> = clock.clone();
    let start_ns = clock.now_ns();
    let source: Rc<dyn SimEventSource> = host.clone();
    let wakeup = host.submit_wakeup();
    let outcome = drive_real_with_source(source, wakeup, move |_handle| async move {
        let run = run_paced_with_backend(
            clock_for_body,
            start_ns,
            sink,
            vec!["dynosim".to_string()],
            workload,
            pattern,
            rate,
            smoothness,
            concurrency,
            prefill_concurrency,
            stop,
            seed,
            adaptive,
            ancillary,
        )
        .await;
        *result_for_body.borrow_mut() = Some(run);
    })?;
    anyhow::ensure!(
        !outcome.deadlocked,
        "Dynamo online (wall-clock in-process) run deadlocked"
    );
    let mut aiperf = result
        .borrow_mut()
        .take()
        .context("Dynamo online driver exited without a paced result")??;
    let dynamo = host.take_report_at(aiperf.performance.throughput.wall_time_ms);
    let (performance, dynamo, parity) =
        finish_shared_metrics_enforcing(aiperf.performance, dynamo, false)?;
    aiperf.performance = performance;
    inject_dynamo_goodput(&mut aiperf.metrics, &dynamo);
    Ok(OfflineRunReport {
        aiperf,
        dynamo,
        parity,
    })
}

fn standard_trace_driver(
    trace: Trace,
    format: OfflineTraceFormat,
    engine_block_size: usize,
    replay_concurrency: Option<usize>,
    arrival_speedup_ratio: f64,
) -> Result<WorkloadDriver> {
    let trace = trace
        .normalize_session_starts()?
        .speed_up_timing(arrival_speedup_ratio)?;
    match (format, replay_concurrency) {
        (OfflineTraceFormat::MooncakeDelta, Some(limit)) => trace
            .into_delta_accumulating_concurrency_driver_with_block_size(engine_block_size, limit),
        (OfflineTraceFormat::MooncakeDelta, None) => {
            trace.into_delta_accumulating_trace_driver_with_block_size(engine_block_size)
        }
        (_, Some(limit)) => trace.into_concurrency_driver_with_block_size(engine_block_size, limit),
        (_, None) => trace.into_trace_driver_with_block_size(engine_block_size),
    }
}

fn agentic_trace_driver(
    trace: AgenticTrace,
    engine_block_size: usize,
    arrival_speedup_ratio: f64,
) -> Result<WorkloadDriver> {
    trace
        .normalize_starts()
        .speed_up_timing(arrival_speedup_ratio)?
        .into_trace_driver_with_block_size(engine_block_size)
}

fn load_trace_driver(
    engine_config: &OfflineEngineConfig,
    config: &OfflineTraceConfig,
) -> Result<WorkloadDriver> {
    anyhow::ensure!(
        config.arrival_speedup_ratio.is_finite() && config.arrival_speedup_ratio > 0.0,
        "--arrival-speedup-ratio must be a finite positive number"
    );
    if let Some(limit) = config.replay_concurrency {
        anyhow::ensure!(limit > 0, "--replay-concurrency must be positive");
    }
    if let Some(cap) = config.max_sim_time_ms {
        anyhow::ensure!(
            cap.is_finite() && cap >= 0.0,
            "--max-sim-time-seconds must be finite and non-negative"
        );
    }
    if config.format == OfflineTraceFormat::AppliedComputeAgentic {
        anyhow::ensure!(
            config.replay_concurrency.is_some(),
            "--trace-format=applied_compute_agentic requires --replay-concurrency"
        );
    }
    if matches!(
        config.format,
        OfflineTraceFormat::AgenticMooncake | OfflineTraceFormat::Dynamo
    ) && config.replay_concurrency.is_some()
    {
        anyhow::ensure!(
            config.format != OfflineTraceFormat::AgenticMooncake,
            "agentic_mooncake trace format is not supported with --replay-concurrency"
        );
    }
    if engine_config.topology == OfflineTopology::Disaggregated {
        anyhow::ensure!(
            config.format != OfflineTraceFormat::MooncakeDelta,
            "mooncake-delta trace format is not supported for disaggregated replay"
        );
        anyhow::ensure!(
            config.format != OfflineTraceFormat::AgenticMooncake,
            "agentic_mooncake trace format is not supported for disaggregated replay"
        );
    }

    let engine_block_size = engine_config.trace_engine_block_size()?;
    let format: TraceFileFormat = config.format.into();
    dynamo_mocker::loadgen::validate_trace_files(format, &config.paths)?;
    match config.format {
        OfflineTraceFormat::Mooncake | OfflineTraceFormat::MooncakeDelta => {
            let block_size = config.trace_block_size.unwrap_or(512);
            let trace = Trace::from_mooncake(&config.paths[0], block_size)?;
            standard_trace_driver(
                trace,
                config.format,
                engine_block_size,
                config.replay_concurrency,
                config.arrival_speedup_ratio,
            )
        }
        OfflineTraceFormat::AppliedComputeAgentic => {
            let block_size = config.trace_block_size.unwrap_or(512);
            let trace = Trace::from_applied_compute_agentic(
                &config.paths[0],
                block_size,
                config.shared_prefix_ratio,
                config.num_prefix_groups,
            )?;
            standard_trace_driver(
                trace,
                config.format,
                engine_block_size,
                config.replay_concurrency,
                config.arrival_speedup_ratio,
            )
        }
        OfflineTraceFormat::AgenticMooncake => {
            let block_size = config.trace_block_size.unwrap_or(512);
            agentic_trace_driver(
                AgenticTrace::from_agentic_mooncake(&config.paths[0], block_size)?,
                engine_block_size,
                config.arrival_speedup_ratio,
            )
        }
        OfflineTraceFormat::Dynamo => match DynamoRequestTrace::from_request_trace_files(
            &config.paths,
            config.trace_block_size,
        )? {
            DynamoRequestTrace::Standard(trace) => standard_trace_driver(
                trace,
                config.format,
                engine_block_size,
                config.replay_concurrency,
                config.arrival_speedup_ratio,
            ),
            DynamoRequestTrace::Agentic(trace) => {
                anyhow::ensure!(
                    engine_config.topology != OfflineTopology::Disaggregated,
                    "agentic Dynamo request traces are not supported for disaggregated replay"
                );
                anyhow::ensure!(
                    config.replay_concurrency.is_none(),
                    "agentic Dynamo request traces are not supported with --replay-concurrency"
                );
                agentic_trace_driver(trace, engine_block_size, config.arrival_speedup_ratio)
            }
        },
    }
}

fn load_worker_artifact_trace(config: &OfflineTraceConfig) -> Result<Trace> {
    let trace = match config.format {
        OfflineTraceFormat::Mooncake => {
            Trace::from_mooncake(&config.paths[0], config.trace_block_size.unwrap_or(512))?
        }
        OfflineTraceFormat::Dynamo => match DynamoRequestTrace::from_request_trace_files(
            &config.paths,
            config.trace_block_size,
        )? {
            DynamoRequestTrace::Standard(trace) => trace,
            DynamoRequestTrace::Agentic(_) => {
                anyhow::bail!("timed worker artifacts require a standard, non-agentic Dynamo trace")
            }
        },
        _ => anyhow::bail!(
            "timed worker artifacts support cumulative mooncake or standard Dynamo traces"
        ),
    };
    trace
        .normalize_session_starts()?
        .speed_up_timing(config.arrival_speedup_ratio)
}

/// Generate and write Dynamo's exact single-worker timed request, output, and
/// KV-event artifacts for a native AIPerf trace configuration.
pub fn write_dynamo_worker_artifacts_json(
    engine_config: &OfflineEngineConfig,
    trace_config: &OfflineTraceConfig,
    visibility: Option<OfflineKvEventVisibility>,
    path: impl AsRef<Path>,
) -> Result<()> {
    anyhow::ensure!(
        engine_config.topology == OfflineTopology::Single,
        "timed worker artifacts require --offline-topology single"
    );
    anyhow::ensure!(
        trace_config.replay_concurrency.is_none(),
        "timed worker artifacts require authored trace timing, not --replay-concurrency"
    );
    let (args, _) = OfflineEngineConfig::configure_aic(engine_config.engine_args()?)?;
    let trace = load_worker_artifact_trace(trace_config)?;
    let artifacts = match visibility {
        Some(visibility) => generate_trace_worker_artifacts_offline_with_kv_event_visibility(
            args,
            trace,
            visibility.into(),
        )?,
        None => generate_trace_worker_artifacts_offline(args, trace)?,
    };
    let requests = artifacts
        .requests
        .into_iter()
        .map(|request| {
            serde_json::json!({
                "uuid": request.uuid,
                "timestamp_us": request.timestamp_us,
                "scheduled_ready_at_ms": request.scheduled_ready_at_ms,
                "input_length": request.input_length,
                "output_length": request.output_length,
                "local_block_hashes": request
                    .replay_hashes
                    .local_block_hashes
                    .into_iter()
                    .map(|hash| hash.0)
                    .collect::<Vec<_>>(),
                "sequence_hashes": request.replay_hashes.sequence_hashes,
            })
        })
        .collect::<Vec<_>>();
    let output_signals = artifacts
        .output_signals
        .into_iter()
        .map(|output| {
            serde_json::json!({
                "timestamp_us": output.timestamp_us,
                "signal": output.signal,
            })
        })
        .collect::<Vec<_>>();
    let kv_events = artifacts
        .kv_events
        .into_iter()
        .map(|event| {
            serde_json::json!({
                "timestamp_us": event.timestamp_us,
                "storage_tier": event.storage_tier,
                "event": event.event,
            })
        })
        .collect::<Vec<_>>();
    let payload = serde_json::json!({
        "timed_requests": requests,
        "timed_output_signals": output_signals,
        "timed_kv_events": kv_events,
    });
    let path = path.as_ref();
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating worker-artifact directory {}", parent.display()))?;
    }
    std::fs::write(path, serde_json::to_string_pretty(&payload)?)
        .with_context(|| format!("writing worker artifacts {}", path.display()))?;
    Ok(())
}

#[derive(Debug)]
struct TraceClientState {
    start_ns: i64,
    prompt_tokens: usize,
    observed_output_tokens: usize,
    admitted: bool,
}

const OFFLINE_REPORT_DECIMAL_SCALE: f64 = 1_000_000.0;

fn canonicalize_offline_float(value: &mut f64) {
    if !value.is_finite() {
        return;
    }
    let scaled = *value * OFFLINE_REPORT_DECIMAL_SCALE;
    if scaled.is_finite() {
        *value = scaled.round() / OFFLINE_REPORT_DECIMAL_SCALE;
    }
    // Collapse -0.0 to +0.0 so byte-exact offline report output never carries a negative zero.
    if *value == 0.0 {
        *value = 0.0;
    }
}

fn canonicalize_offline_distribution(distribution: &mut DynamoDistributionStats) {
    canonicalize_offline_float(&mut distribution.mean_ms);
    canonicalize_offline_float(&mut distribution.min_ms);
    canonicalize_offline_float(&mut distribution.max_ms);
    canonicalize_offline_float(&mut distribution.median_ms);
    canonicalize_offline_float(&mut distribution.p75_ms);
    canonicalize_offline_float(&mut distribution.p90_ms);
    canonicalize_offline_float(&mut distribution.p95_ms);
    canonicalize_offline_float(&mut distribution.p99_ms);
    canonicalize_offline_float(&mut distribution.std_ms);
}

/// Normalize only the frontend report boundary for reproducible offline JSON.
///
/// Dynamo remains the sole simulation and aggregation implementation. Offline
/// host execution time is measured externally, while the serialized report
/// uses the simulated duration. Nanosecond-scale decimal normalization removes
/// floating reduction-order noise without changing routing or mocker behavior.
pub fn canonicalize_offline_report(mut report: DynamoSimulationReport) -> DynamoSimulationReport {
    report.throughput.wall_time_ms = report.throughput.duration_ms;
    canonicalize_offline_float(&mut report.throughput.duration_ms);
    canonicalize_offline_float(&mut report.throughput.wall_time_ms);
    canonicalize_offline_float(&mut report.throughput.request_throughput_rps);
    canonicalize_offline_float(&mut report.throughput.input_throughput_tok_s);
    canonicalize_offline_float(&mut report.throughput.output_throughput_tok_s);
    canonicalize_offline_float(&mut report.throughput.total_throughput_tok_s);
    canonicalize_offline_float(&mut report.throughput.prefill_worker_seconds);
    canonicalize_offline_float(&mut report.throughput.decode_worker_seconds);
    canonicalize_offline_float(&mut report.throughput.gpu_hours);
    canonicalize_offline_float(&mut report.prefix_cache_reused_ratio);
    canonicalize_offline_float(&mut report.first_admission_prefix_cache_reused_ratio);
    canonicalize_offline_distribution(&mut report.latency.ttft);
    canonicalize_offline_distribution(&mut report.latency.ttst);
    canonicalize_offline_distribution(&mut report.latency.tpot);
    canonicalize_offline_distribution(&mut report.latency.itl.distribution);
    canonicalize_offline_float(&mut report.latency.itl.max_ms);
    canonicalize_offline_distribution(&mut report.latency.e2e);
    canonicalize_offline_distribution(&mut report.latency.output_token_throughput_per_user);
    if let Some(goodput) = &mut report.goodput {
        canonicalize_offline_float(&mut goodput.request_throughput_rps);
        canonicalize_offline_float(&mut goodput.output_throughput_tok_s);
    }
    report
}

/// Run canonical trace replay through the exact static Dynamo harness used by
/// `python -m dynamo.replay`.
///
/// This path intentionally owns no AIPerf observer, compatibility collector,
/// or native-metrics column store. Frontends lower configuration into Dynamo's
/// shared [`DynamoTraceReplayConfig`], so scheduler events, floating-point time,
/// aggregation order, and report finalization execute once in the owning crate.
pub fn run_canonical_trace_offline(
    engine_config: OfflineEngineConfig,
    trace_config: OfflineTraceConfig,
) -> Result<DynamoSimulationReport> {
    let router_config = engine_config.replay_router_config()?;
    let requested_router_mode: ReplayRouterMode = engine_config.router_mode.into();
    let (engines, prefill_load_estimator, router_mode) = match engine_config.topology {
        OfflineTopology::Single => {
            let (args, estimator) =
                OfflineEngineConfig::configure_aic(engine_config.engine_args()?)?;
            let estimator =
                engine_config.validate_prefill_load_estimator(router_config.as_ref(), estimator)?;
            (
                OfflineTraceReplayEngines::Aggregated {
                    args,
                    num_workers: 1,
                },
                estimator,
                ReplayRouterMode::RoundRobin,
            )
        }
        OfflineTopology::Aggregated => {
            anyhow::ensure!(
                engine_config.workers > 0,
                "offline aggregate workers must be positive"
            );
            let (args, estimator) =
                OfflineEngineConfig::configure_aic(engine_config.engine_args()?)?;
            let estimator =
                engine_config.validate_prefill_load_estimator(router_config.as_ref(), estimator)?;
            (
                OfflineTraceReplayEngines::Aggregated {
                    args,
                    num_workers: engine_config.workers,
                },
                estimator,
                requested_router_mode,
            )
        }
        OfflineTopology::Disaggregated => {
            anyhow::ensure!(
                engine_config.prefill_workers > 0,
                "offline prefill workers must be positive"
            );
            anyhow::ensure!(
                engine_config.decode_workers > 0,
                "offline decode workers must be positive"
            );
            let (prefill_args, decode_args) = engine_config.disaggregated_engine_args()?;
            let (prefill_args, estimator) = OfflineEngineConfig::configure_aic(prefill_args)?;
            let (decode_args, _) = OfflineEngineConfig::configure_aic(decode_args)?;
            let estimator =
                engine_config.validate_prefill_load_estimator(router_config.as_ref(), estimator)?;
            (
                OfflineTraceReplayEngines::Disaggregated(OfflineDisaggReplayConfig {
                    prefill_args,
                    decode_args,
                    num_prefill_workers: engine_config.prefill_workers,
                    num_decode_workers: engine_config.decode_workers,
                }),
                estimator,
                requested_router_mode,
            )
        }
    };

    let report = simulate_offline_trace_files(DynamoTraceReplayConfig {
        engines,
        router_config,
        prefill_load_estimator,
        trace_files: trace_config.paths,
        trace_block_size: trace_config.trace_block_size,
        replay_concurrency: trace_config.replay_concurrency,
        router_mode,
        arrival_speedup_ratio: trace_config.arrival_speedup_ratio,
        trace_format: trace_config.format.into(),
        trace_shared_prefix_ratio: trace_config.shared_prefix_ratio,
        trace_num_prefix_groups: trace_config.num_prefix_groups,
        record_per_request: engine_config.capture_per_request,
        max_sim_time_ms: trace_config.max_sim_time_ms,
        sla: engine_config.sla,
    })?;
    Ok(canonicalize_offline_report(report))
}

/// Replay any canonical Dynamo trace format through AIPerf's native observer
/// and report stack while the steppable mocker owns scheduling/KV semantics.
pub fn run_trace_offline(
    engine_config: OfflineEngineConfig,
    trace_config: OfflineTraceConfig,
) -> Result<OfflineRunReport> {
    let mut driver = load_trace_driver(&engine_config, &trace_config)?;
    let mut engine = engine_config.build()?;
    let clock = Rc::new(SimClock::new());
    let collector = Rc::new(CollectorObserver::new(true));
    let native_metrics = Rc::new(NativeMetricsObserver::new(
        clock.clone(),
        0,
        MetricsConfig::default(),
    ));
    let delegates: Vec<Rc<dyn RequestObserver>> = vec![collector.clone(), native_metrics.clone()];
    let observer: Rc<dyn RequestObserver> = Rc::new(ObserverTee::new(delegates));
    let mut clients: HashMap<Uuid, TraceClientState> = HashMap::new();
    let mut now_ms = 0.0_f64;
    let mut same_instant_control_steps = 0_usize;
    let cap_ms = trace_config.max_sim_time_ms.unwrap_or(f64::INFINITY);

    while !driver.is_drained() || !engine.is_idle() {
        let next_ready = driver.next_ready_time_ms();
        let next_engine = engine
            .next_event_ms()
            .or_else(|| (!engine.is_idle()).then_some(engine.now_ms()));
        let next_ms = match (next_ready, next_engine) {
            (Some(ready), Some(engine)) => ready.min(engine),
            (Some(ready), None) => ready,
            (None, Some(engine)) => engine,
            (None, None) => anyhow::bail!(
                "Dynamo trace replay deadlocked with {} in-flight requests",
                engine.in_flight()
            ),
        };
        if next_ms > cap_ms {
            break;
        }
        anyhow::ensure!(
            next_ms.is_finite() && next_ms >= now_ms,
            "Dynamo trace clock regressed from {now_ms}ms to {next_ms}ms"
        );
        now_ms = next_ms;
        engine.advance_now_ms(now_ms);
        clock.advance_to(ms_to_ns(now_ms).map_err(anyhow::Error::msg)?);

        let ready = driver.pop_ready(now_ms, usize::MAX);
        let submitted = !ready.is_empty();
        for ready in ready {
            let uuid = ready.request_uuid;
            let prompt_tokens = ready.request.tokens.len();
            let requested_output_tokens = ready.request.max_output_tokens;
            native_metrics.register_metadata(
                uuid,
                RequestMetricMetadata {
                    conversation_id: Some(ready.session_id),
                    turn_index: u32::try_from(ready.turn_index).unwrap_or(u32::MAX),
                    correlation_id: ready.replay_key,
                    dimensions: InferenceDimensions {
                        endpoint_url: Some("dynosim://offline".to_string()),
                        model: None,
                    },
                    ..RequestMetricMetadata::default()
                },
            );
            observer.on_arrival(uuid, now_ms, prompt_tokens, requested_output_tokens);
            let assigned = engine.submit(ready.request)?;
            anyhow::ensure!(assigned == uuid, "Dynamo changed trace request id {uuid}");
            clients.insert(
                uuid,
                TraceClientState {
                    start_ns: ms_to_ns(now_ms).map_err(anyhow::Error::msg)?,
                    prompt_tokens,
                    observed_output_tokens: 0,
                    admitted: false,
                },
            );
        }

        let next_external = driver.next_ready_time_ms().unwrap_or(f64::INFINITY);
        let deadline_ms = next_external.min(cap_ms);
        let engine_due = engine
            .next_event_ms()
            .or_else(|| (!engine.is_idle()).then_some(engine.now_ms()))
            .is_some_and(|event_ms| event_ms <= deadline_ms);
        let mut made_progress = submitted;
        if engine_due {
            let before_ms = now_ms;
            let before_in_flight = engine.in_flight();
            let before_event_ms = engine.next_event_ms();
            let outcome = engine.step_until(deadline_ms)?;
            anyhow::ensure!(
                outcome.end_ms >= before_ms && outcome.end_ms <= deadline_ms,
                "Dynamo trace step escaped [{before_ms}, {deadline_ms}] at {}ms",
                outcome.end_ms
            );
            now_ms = outcome.end_ms;
            clock.advance_to(ms_to_ns(now_ms).map_err(anyhow::Error::msg)?);
            let after_in_flight = engine.in_flight();
            let after_event_ms = engine.next_event_ms();
            made_progress |= now_ms > before_ms
                || !outcome.events.is_empty()
                || after_in_flight != before_in_flight
                || after_event_ms != before_event_ms;

            for event in outcome.events {
                let Some(client) = clients.get_mut(&event.uuid) else {
                    continue;
                };
                if !client.admitted
                    && let Some((at_ms, reused_input_tokens)) = engine.request_admission(event.uuid)
                {
                    observer.on_admit(event.uuid, at_ms, reused_input_tokens);
                    client.admitted = true;
                }
                if let Some(token_id) = event.token_id {
                    driver.on_output_token(event.uuid, token_id)?;
                    client.observed_output_tokens += 1;
                    observer.on_classified_token(event.uuid, now_ms, ObservedTokenKind::Output);
                }
                let Some(status) = event.terminal_status else {
                    continue;
                };
                let terminal = map_terminal_status(status);
                let completion_tokens = engine
                    .actual_output_length(event.uuid)
                    .unwrap_or(client.observed_output_tokens);
                observer.on_usage(
                    event.uuid,
                    ObservedUsage {
                        prompt_tokens: Some(client.prompt_tokens),
                        completion_tokens: Some(completion_tokens),
                        ..ObservedUsage::default()
                    },
                );
                observer.on_terminal(event.uuid, terminal);
                native_metrics.record_response(
                    event.uuid,
                    NativeResponseMetadata {
                        start_ns: Some(client.start_ns),
                        end_ns: Some(ms_to_ns(now_ms).map_err(anyhow::Error::msg)?),
                        prompt_tokens: u64::try_from(client.prompt_tokens).ok(),
                        completion_tokens: u64::try_from(completion_tokens).ok(),
                        http: RequestTrace::default(),
                    },
                );
                match status {
                    DynamoTerminalStatus::Completed => driver.on_complete(event.uuid, now_ms)?,
                    DynamoTerminalStatus::Rejected => {
                        driver.on_terminal(event.uuid, now_ms, true)?
                    }
                    DynamoTerminalStatus::Canceled | DynamoTerminalStatus::Failed => {
                        driver.release_cap_slot(event.uuid, now_ms)
                    }
                }
                clients.remove(&event.uuid);
            }
        }
        if made_progress || next_ms == cap_ms {
            same_instant_control_steps = 0;
        } else {
            same_instant_control_steps = same_instant_control_steps.saturating_add(1);
            anyhow::ensure!(
                same_instant_control_steps < 1_024,
                "Dynamo trace replay exhausted same-instant control progress at {next_ms}ms: engine_now={} in_flight={} idle={} next_engine={:?} next_ready={:?}",
                engine.now_ms(),
                engine.in_flight(),
                engine.is_idle(),
                engine.next_event_ms(),
                driver.next_ready_time_ms(),
            );
        }
    }

    let dynamo = engine.take_report(now_ms);
    let performance = collector.finish(now_ms);
    let (performance, dynamo, parity) = finish_shared_metrics(performance, dynamo)?;
    let mut metrics = native_metrics.finish();
    inject_dynamo_goodput(&mut metrics, &dynamo);
    Ok(OfflineRunReport {
        aiperf: OnlineRunReport {
            performance,
            metrics,
        },
        dynamo,
        parity,
    })
}

struct DynamoGraphSink {
    backend: Rc<DynosimSink>,
    observer: Rc<dyn RequestObserver>,
    native_metrics: Rc<NativeMetricsObserver>,
    phase: MetricsPhase,
    events: Rc<dyn OfflineGraphEventSink>,
    input_tokens_by_node: Option<Arc<HashMap<String, usize>>>,
    trace_id: Option<String>,
    max_tokens: usize,
    next_uuid: Rc<Cell<u128>>,
}

impl DynamoGraphSink {
    async fn dispatch_node(
        &self,
        node_id: &str,
        messages: Vec<Bytes>,
        max_tokens: Option<usize>,
        cancel_after_ns: Option<i64>,
        on_first_token: &dyn Fn(),
    ) -> Result<GraphReply<GraphMessage>> {
        let request_ordinal = self.next_uuid.get();
        let uuid = Uuid::from_u128(request_ordinal);
        self.next_uuid.set(request_ordinal.saturating_add(1));
        let (mut tokens, input_tokens) = match self
            .input_tokens_by_node
            .as_ref()
            .and_then(|tokens| tokens.get(node_id))
            .copied()
        {
            Some(input_tokens) => (
                self.backend
                    .graph_encoder
                    .encode_graph_messages(&messages, input_tokens)?,
                input_tokens,
            ),
            None => {
                let tokens = self
                    .backend
                    .graph_encoder
                    .encode_graph_messages_exact(&messages)?;
                let input_tokens = tokens.len().max(1);
                (tokens, input_tokens)
            }
        };
        if tokens.is_empty() {
            tokens.push(0);
        }
        let output_tokens = max_tokens.unwrap_or(self.max_tokens).max(1);
        let now_ns = self.backend.clock.now_ns();
        let (correlation_id, conversation_id) = self.trace_id.as_ref().map_or_else(
            || (uuid.to_string(), node_id.to_string()),
            |trace_id| (format!("{trace_id}/{node_id}/{uuid}"), trace_id.to_string()),
        );
        self.native_metrics.register_metadata(
            uuid,
            RequestMetricMetadata {
                request_index: usize::try_from(request_ordinal.saturating_sub(1)).ok(),
                phase: self.phase,
                correlation_id: Some(correlation_id),
                conversation_id: Some(conversation_id.clone()),
                dimensions: InferenceDimensions {
                    endpoint_url: Some("dynosim://offline".to_string()),
                    model: Some(self.backend.model.clone()),
                },
                ..RequestMetricMetadata::default()
            },
        );
        self.observer.on_arrival(
            uuid,
            self.backend.relative_ms(now_ns),
            input_tokens,
            output_tokens,
        );

        let record_trace_id = self
            .trace_id
            .clone()
            .unwrap_or_else(|| conversation_id.clone());
        let first_token_seen = Cell::new(false);
        let first_token_error = RefCell::new(None);
        let first_token = |_delta_ns| {
            if !first_token_seen.replace(true) {
                if let Err(error) = self.events.first_token(uuid, &record_trace_id) {
                    *first_token_error.borrow_mut() = Some(error);
                }
                on_first_token();
            }
        };
        let result = self
            .backend
            .dispatch_tokens(
                uuid,
                tokens,
                output_tokens,
                cancel_after_ns,
                self.observer.as_ref(),
                &first_token,
            )
            .await;
        if let Some(error) = first_token_error.borrow_mut().take() {
            return Err(error);
        }
        let result = match result {
            Ok(result) => result,
            Err(error) => {
                self.observer
                    .on_terminal(uuid, ReplayTerminalStatus::Failed);
                self.emit_record(uuid, request_ordinal, String::new())?;
                tracing::warn!(%uuid, %node_id, %error, "offline graph dispatch failed");
                return Ok(GraphReply::failed());
            }
        };

        self.native_metrics.record_response(
            uuid,
            NativeResponseMetadata {
                start_ns: Some(result.start_ns),
                end_ns: Some(result.end_ns),
                prompt_tokens: result.prompt_tokens.map(u64::from),
                completion_tokens: result.completion_tokens.map(u64::from),
                http: result.http,
            },
        );
        self.emit_record(uuid, request_ordinal, result.response_text.clone())?;
        Ok(match result.terminal {
            ReplayTerminalStatus::Completed => GraphReply::from_text(result.response_text),
            ReplayTerminalStatus::Canceled => GraphReply::cancelled(),
            ReplayTerminalStatus::Rejected | ReplayTerminalStatus::Failed => GraphReply::failed(),
        })
    }

    fn emit_record(&self, uuid: Uuid, request_ordinal: u128, response_text: String) -> Result<()> {
        let ordinal = u64::try_from(request_ordinal.saturating_sub(1)).unwrap_or(u64::MAX);
        let ingest = self
            .native_metrics
            .snapshot_record(uuid, ordinal)
            .with_context(|| format!("offline graph request {uuid} has no native metric record"))?;
        let trace_id = self
            .trace_id
            .clone()
            .unwrap_or_else(|| ingest.correlation_id.clone());
        self.events.record(OfflineGraphRequestRecord {
            uuid,
            trace_id,
            response_text,
            ingest,
        })
    }
}

#[async_trait(?Send)]
impl GraphSink<GraphMessage> for DynamoGraphSink {
    async fn dispatch(
        &self,
        node_id: &str,
        messages: Vec<Bytes>,
        max_tokens: Option<usize>,
        on_first_token: &dyn Fn(),
    ) -> Result<GraphReply<GraphMessage>> {
        self.dispatch_node(node_id, messages, max_tokens, None, on_first_token)
            .await
    }

    async fn dispatch_with_options(
        &self,
        node_id: &str,
        messages: Vec<Bytes>,
        max_tokens: Option<usize>,
        options: GraphDispatchOptions,
        on_first_token: &dyn Fn(),
    ) -> Result<GraphReply<GraphMessage>> {
        self.dispatch_node(
            node_id,
            messages,
            max_tokens,
            options.cancel_after_ns,
            on_first_token,
        )
        .await
    }
}

/// Run the generated multi-turn Graph-IR workload against the in-process
/// Dynamo engine. All trace lanes share one `current_thread`/`LocalSet`, one
/// [`SimClock`], and one engine so batching and KV contention are global.
pub fn run_graph_offline(
    engine_config: OfflineEngineConfig,
    config: BenchConfig,
) -> Result<OfflineGraphReport> {
    let turns = config.turns.max(1);
    let (pool, graph, input_tokens_by_node) = build_workload(turns);
    let graph = Rc::new(graph);
    let store: Arc<dyn SegmentStore> = Arc::new(pool);
    let materializer = Rc::new(SegmentItemsMaterializer::new(store));

    let clock = Rc::new(SimClock::new());
    let clock_trait: Rc<dyn Clock> = clock.clone();
    let start_ns = clock.now_ns();
    let host = EngineHost::new(clock.clone(), &engine_config)?;
    let backend = DynosimSink::new(host.clone(), clock.clone(), config.model.clone());
    let native_metrics = Rc::new(NativeMetricsObserver::new(
        clock_trait,
        start_ns,
        MetricsConfig::default(),
    ));
    let collector = Rc::new(CollectorObserver::new(false));
    let observer: Rc<dyn RequestObserver> = Rc::new(ObserverTee::new(vec![
        collector.clone(),
        native_metrics.clone(),
    ]));
    let sink: Rc<dyn GraphSink<GraphMessage>> = Rc::new(DynamoGraphSink {
        backend,
        observer,
        native_metrics: native_metrics.clone(),
        phase: MetricsPhase::Profiling,
        events: Rc::new(DiscardOfflineGraphEvents),
        input_tokens_by_node: Some(Arc::new(input_tokens_by_node)),
        trace_id: None,
        max_tokens: config.max_tokens.max(1),
        next_uuid: Rc::new(Cell::new(1)),
    });

    let next_trace = Rc::new(Cell::new(0_usize));
    let lane_count = config
        .workers
        .max(1)
        .saturating_mul(config.concurrency.max(1));
    let instances = config.instances;
    let duration_ns = config.max_duration_ns;
    let source: Rc<dyn SimEventSource> = host.clone();
    let driver_clock = clock.clone();
    let outcome = drive_sim_with_source(clock.clone(), source, move |_driver_handle| async move {
        let mut lanes = Vec::with_capacity(lane_count);
        for _ in 0..lane_count {
            let graph = graph.clone();
            let materializer = materializer.clone();
            let sink = sink.clone();
            let clock = driver_clock.clone();
            let next_trace = next_trace.clone();
            lanes.push(tokio::task::spawn_local(async move {
                loop {
                    if duration_ns.is_some_and(|duration| clock.now_ns() >= duration) {
                        break;
                    }
                    let index = next_trace.get();
                    next_trace.set(index.saturating_add(1));
                    if index >= instances {
                        break;
                    }
                    let handle = crate::graph::runtime::Handle::new(clock.clone());
                    let executor = match TraceExecutor::new(
                        graph.clone(),
                        materializer.clone(),
                        sink.clone(),
                        handle.clone(),
                        ExecutorFlags::default(),
                    ) {
                        Ok(executor) => executor,
                        Err(error) => {
                            tracing::warn!(%error, "failed to construct offline graph trace");
                            continue;
                        }
                    };
                    let trace = TraceRecord {
                        id: format!("offline-{index}"),
                        graph_ref: None,
                        initial_state: Default::default(),
                    };
                    match executor.build_context(trace) {
                        Ok(context) => {
                            executor.schedule_entries(&context);
                            handle.wait_idle().await;
                        }
                        Err(error) => {
                            tracing::warn!(%error, "failed to build offline graph context");
                        }
                    }
                }
            }));
        }
        for lane in lanes {
            if let Err(error) = lane.await {
                tracing::warn!(%error, "offline graph lane failed");
            }
        }
    })?;
    anyhow::ensure!(!outcome.deadlocked, "Dynamo offline graph run deadlocked");

    let wall_ms = clock.now_ns().saturating_sub(start_ns) as f64 / NS_PER_MS;
    let dynamo = host.take_report_at(wall_ms);
    let performance = collector.finish(wall_ms);
    let (performance, dynamo, parity) = finish_shared_metrics(performance, dynamo)?;
    let mut graph_report = GraphRpsReport {
        completed: u64::try_from(performance.request_counts.completed_requests).unwrap_or(u64::MAX),
        errors: u64::try_from(
            performance
                .request_counts
                .num_requests
                .saturating_sub(performance.request_counts.completed_requests),
        )
        .unwrap_or(u64::MAX),
        output_tokens: u64::try_from(performance.request_counts.total_output_tokens)
            .unwrap_or(u64::MAX),
        wall_secs: performance.throughput.wall_time_ms / 1_000.0,
        ttft_p50_ms: performance.latency.ttft.median_ms,
        ttft_p90_ms: performance.latency.ttft.p90_ms,
        ttft_p99_ms: performance.latency.ttft.p99_ms,
        ttft_mean_ms: performance.latency.ttft.mean_ms,
        native_metrics: native_metrics.finish(),
    };
    inject_dynamo_goodput(&mut graph_report.native_metrics, &dynamo);
    Ok(OfflineGraphReport {
        aiperf: graph_report,
        performance,
        dynamo,
        parity,
    })
}

struct DynamoDirectGraphBackend {
    backend: Rc<DynosimSink>,
    observer: Rc<dyn RequestObserver>,
    native_metrics: Rc<NativeMetricsObserver>,
    phase: MetricsPhase,
    events: Rc<dyn OfflineGraphEventSink>,
    materializer: Rc<SegmentItemsMaterializer>,
    node_policy: Option<Rc<dyn NodeDispatchPolicy>>,
    node_failure: Rc<dyn NodeFailurePolicy>,
    prefill_slots: Option<Rc<SlotPool>>,
    max_tokens: usize,
    next_uuid: Rc<Cell<u128>>,
    cancelled: Cell<bool>,
    active: RefCell<Vec<Weak<LocalGraphTraceExecutionBackend<GraphMessage>>>>,
}

#[async_trait(?Send)]
impl TracePlacement for DynamoDirectGraphBackend {
    async fn execute_trace(
        &self,
        plan: GraphTracePlan,
    ) -> Result<(), crate::graph::errors::TraceError> {
        let trace_id = plan.trace.id.clone();
        if self.cancelled.get() {
            return Err(crate::graph::errors::TraceError::Cancelled(format!(
                "offline graph trace {trace_id:?} was cancelled by its phase backend"
            )));
        }
        let input_tokens_by_node = plan
            .graph
            .nodes
            .iter()
            .filter_map(|(node_id, node)| {
                node.metadata
                    .get("input_tokens")
                    .and_then(serde_json::Value::as_u64)
                    .map(|tokens| {
                        usize::try_from(tokens)
                            .map(|tokens| (node_id.clone(), tokens))
                            .map_err(|_| {
                                crate::graph::errors::TraceError::Other(format!(
                                    "graph node {node_id:?} input_tokens exceeds usize"
                                ))
                            })
                    })
            })
            .collect::<std::result::Result<HashMap<_, _>, _>>()?;
        let sink: Rc<dyn GraphSink<GraphMessage>> = Rc::new(DynamoGraphSink {
            backend: self.backend.clone(),
            observer: self.observer.clone(),
            native_metrics: self.native_metrics.clone(),
            phase: self.phase,
            events: self.events.clone(),
            input_tokens_by_node: Some(Arc::new(input_tokens_by_node)),
            trace_id: Some(trace_id.clone()),
            max_tokens: self.max_tokens,
            next_uuid: self.next_uuid.clone(),
        });
        let mut local = LocalGraphTraceExecutionBackend::new(
            self.backend.clock.clone(),
            self.materializer.clone(),
            sink,
        )
        .with_node_failure(self.node_failure.clone());
        if let Some(policy) = &self.node_policy {
            local = local.with_node_policy(policy.clone());
        }
        let local = Rc::new(local);
        self.active.borrow_mut().push(Rc::downgrade(&local));
        let result = local.execute_trace(plan).await;
        self.active
            .borrow_mut()
            .retain(|active| active.as_ptr() != Rc::as_ptr(&local));
        result
    }

    fn cancel_inflight(&self) -> Result<(), crate::graph::errors::TraceError> {
        self.cancelled.set(true);
        let active = self
            .active
            .borrow()
            .iter()
            .filter_map(Weak::upgrade)
            .collect::<Vec<_>>();
        for backend in active {
            backend.cancel_inflight()?;
        }
        Ok(())
    }

    fn set_prefill_limit(&self, limit: usize) -> Result<(), crate::graph::errors::TraceError> {
        if limit == 0 {
            return Err(crate::graph::errors::TraceError::Other(
                "offline graph prefill limit must be positive".into(),
            ));
        }
        let slots = self.prefill_slots.as_ref().ok_or_else(|| {
            crate::graph::errors::TraceError::Other(
                "offline graph phase does not expose prefill control".into(),
            )
        })?;
        slots.set_limit(limit);
        Ok(())
    }
}

struct DynosimGraphBackendFactory {
    backend: Rc<DynosimSink>,
    observer: Rc<dyn RequestObserver>,
    native_metrics: Rc<NativeMetricsObserver>,
    materializer: Rc<SegmentItemsMaterializer>,
    max_tokens: usize,
    next_uuid: Rc<Cell<u128>>,
}

impl OfflineGraphBackendFactory for DynosimGraphBackendFactory {
    fn create_backend(&self, config: OfflineGraphBackendConfig) -> Result<Rc<dyn TracePlacement>> {
        let prefill_slots = config
            .prefill_concurrency
            .map(|limit| {
                anyhow::ensure!(limit > 0, "offline graph prefill limit must be positive");
                Ok(Rc::new(SlotPool::new(limit)))
            })
            .transpose()?;
        let mut policies = Vec::<Rc<dyn NodeDispatchPolicy>>::new();
        if let Some(slots) = &prefill_slots {
            policies.push(Rc::new(PrefillSlotNodePolicy::new(slots.clone())));
        }
        if let Some(policy) = config.node_policy {
            policies.push(policy);
        }
        let node_policy = match policies.len() {
            0 => None,
            1 => policies.pop(),
            _ => Some(Rc::new(CompositeNodeDispatchPolicy::new(policies)) as Rc<_>),
        };
        Ok(Rc::new(DynamoDirectGraphBackend {
            backend: self.backend.clone(),
            observer: self.observer.clone(),
            native_metrics: self.native_metrics.clone(),
            phase: config.phase,
            events: config.events,
            materializer: self.materializer.clone(),
            node_policy,
            node_failure: config.node_failure,
            prefill_slots,
            max_tokens: self.max_tokens,
            next_uuid: self.next_uuid.clone(),
            cancelled: Cell::new(false),
            active: RefCell::new(Vec::new()),
        }))
    }
}

/// Factory seam for direct Graph-IR workload policy that needs the exact
/// virtual clock and node-execution backend before construction.
pub trait OfflineGraphRunFactory {
    /// Create one policy-composed graph workload. The returned coordinator
    /// owns root selection/admission/arrival/failure policy while `backend`
    /// remains the sole node execution path.
    fn create(
        self: Box<Self>,
        clock: Rc<dyn Clock>,
        backend: Rc<dyn TracePlacement>,
    ) -> Result<GraphWorkload>;
}

impl<F> OfflineGraphRunFactory for F
where
    F: FnOnce(Rc<dyn Clock>, Rc<dyn TracePlacement>) -> Result<GraphWorkload>,
{
    fn create(
        self: Box<Self>,
        clock: Rc<dyn Clock>,
        backend: Rc<dyn TracePlacement>,
    ) -> Result<GraphWorkload> {
        (*self)(clock, backend)
    }
}

/// Execute direct authored Graph-IR plans against the in-process engine.
///
/// The caller injects the ordinary graph root/arrival/admission/failure traits
/// through [`OfflineGraphRunFactory`]. This function supplies the one shared
/// [`SimClock`], steppable engine event source, immutable segment store, node
/// sink, observers, and exact common-summary parity proof. It therefore avoids
/// both the generated benchmark graph and any Dataset/Conversation conversion.
#[allow(clippy::too_many_arguments)]
pub fn run_graph_workload_offline(
    engine_config: OfflineEngineConfig,
    model: String,
    segments: Arc<dyn SegmentStore>,
    default_max_tokens: usize,
    metrics_config: MetricsConfig,
    node_policy: Option<Rc<dyn NodeDispatchPolicy>>,
    node_failure: Rc<dyn NodeFailurePolicy>,
    factory: Box<dyn OfflineGraphRunFactory>,
) -> Result<OfflineDirectGraphReport> {
    let deferred: Box<dyn DeferredOfflineGraphRunFactory> = Box::new(
        move |clock: Rc<dyn Clock>,
              backends: Rc<dyn OfflineGraphBackendFactory>|
              -> Result<DeferredOfflineGraphFuture> {
            let backend = backends.create_backend(OfflineGraphBackendConfig {
                phase: MetricsPhase::Profiling,
                prefill_concurrency: None,
                node_policy,
                node_failure,
                events: Rc::new(DiscardOfflineGraphEvents),
            })?;
            let workload = factory.create(clock, backend)?;
            Ok(Box::pin(async move {
                let workload = workload
                    .execute()
                    .await
                    .map_err(|error| anyhow::anyhow!(error.to_string()))?;
                Ok(OfflineGraphExecution {
                    workload,
                    phases: Vec::new(),
                })
            }))
        },
    );
    run_graph_workload_offline_deferred(
        engine_config,
        model,
        segments,
        default_max_tokens,
        metrics_config,
        deferred,
    )
}

/// Execute a multi-phase direct Graph-IR driver over one offline engine.
///
/// The factory receives the exact run-scoped clock and a phase-local backend
/// factory. It may construct any number of ordinary [`GraphWorkload`] values
/// and drive them through `ClockPhaseOrchestrator`; every node still reaches
/// the same simulator, compatibility collector, and UUID stream.
pub fn run_graph_workload_offline_deferred(
    engine_config: OfflineEngineConfig,
    model: String,
    segments: Arc<dyn SegmentStore>,
    default_max_tokens: usize,
    metrics_config: MetricsConfig,
    factory: Box<dyn DeferredOfflineGraphRunFactory>,
) -> Result<OfflineDirectGraphReport> {
    anyhow::ensure!(
        !model.trim().is_empty(),
        "offline graph model cannot be empty"
    );
    anyhow::ensure!(
        default_max_tokens > 0,
        "offline graph default_max_tokens must be positive"
    );

    let clock = Rc::new(SimClock::new());
    let clock_trait: Rc<dyn Clock> = clock.clone();
    let start_ns = clock.now_ns();
    let host = EngineHost::new(clock.clone(), &engine_config)?;
    let sink = DynosimSink::new(host.clone(), clock.clone(), model);
    let native_metrics = Rc::new(NativeMetricsObserver::new(
        clock_trait.clone(),
        start_ns,
        metrics_config.clone(),
    ));
    let collector = Rc::new(CollectorObserver::new(false));
    let observer: Rc<dyn RequestObserver> = Rc::new(ObserverTee::new(vec![
        collector.clone(),
        native_metrics.clone(),
    ]));
    let backends: Rc<dyn OfflineGraphBackendFactory> = Rc::new(DynosimGraphBackendFactory {
        backend: sink,
        observer,
        native_metrics: native_metrics.clone(),
        materializer: Rc::new(SegmentItemsMaterializer::new(segments)),
        max_tokens: default_max_tokens,
        next_uuid: Rc::new(Cell::new(1)),
    });
    let future = factory.create(clock_trait, backends)?;
    let result = Rc::new(RefCell::new(None));
    let result_for_body = result.clone();
    let source: Rc<dyn SimEventSource> = host.clone();
    let outcome = drive_sim_with_source(clock.clone(), source, move |_handle| async move {
        *result_for_body.borrow_mut() = Some(future.await);
    })?;
    anyhow::ensure!(
        !outcome.deadlocked,
        "Dynamo offline direct graph run deadlocked"
    );
    let execution = result
        .borrow_mut()
        .take()
        .context("Dynamo offline driver exited without a graph phase result")??;
    let wall_ms = clock.now_ns().saturating_sub(start_ns) as f64 / NS_PER_MS;
    let dynamo = host.take_report_at(wall_ms);
    let performance = collector.finish(wall_ms);
    let (performance, dynamo, parity) = finish_shared_metrics(performance, dynamo)?;
    let collection = native_metrics.finish_with_records();
    let mut accumulator = MetricsAccumulator::with_config(metrics_config);
    for (_uuid, record) in &collection.records {
        accumulator.process_record(record);
    }
    let mut profiling_metrics =
        accumulator.export_results(&ExportContext::phase(MetricsPhase::Profiling));
    let warmup_metrics = collection
        .records
        .iter()
        .any(|(_uuid, record)| record.phase == MetricsPhase::Warmup)
        .then(|| accumulator.export_results(&ExportContext::phase(MetricsPhase::Warmup)));
    inject_dynamo_goodput(&mut profiling_metrics, &dynamo);
    Ok(OfflineDirectGraphReport {
        workload: execution.workload,
        performance,
        native_metrics: profiling_metrics,
        warmup_metrics,
        phases: execution.phases,
        dynamo,
        parity,
    })
}

/// Execute a direct Graph-IR workload against the in-process Dynamo engine under
/// the **wall clock** — the real-time in-process online mode (Dynamo's
/// `--replay-mode online`) driven entirely by aiperf's *own* recorded-trace
/// graph flow. The trace is compiled by `aiperf-graph` into a [`GraphWorkload`]
/// and dispatched through the same [`DynosimSink`], materializer, observer,
/// and native accumulator as [`run_graph_workload_offline`]; the mocker's own
/// trace driver is never involved. Only the clock/driver axis differs
/// ([`drive_real_with_source`] vs [`drive_sim_with_source`]).
///
/// Because real timers carry jitter and virtual look-ahead is absent, byte-exact
/// parity with the engine's report is not enforced (see
/// [`finish_shared_metrics_enforcing`]); the shared collector/report path is
/// preserved so metric *identities* still match within the online tolerance.
pub fn run_graph_workload_online(
    engine_config: OfflineEngineConfig,
    model: String,
    segments: Arc<dyn SegmentStore>,
    default_max_tokens: usize,
    metrics_config: MetricsConfig,
    node_policy: Option<Rc<dyn NodeDispatchPolicy>>,
    node_failure: Rc<dyn NodeFailurePolicy>,
    factory: Box<dyn OfflineGraphRunFactory>,
) -> Result<OfflineDirectGraphReport> {
    let deferred: Box<dyn DeferredOfflineGraphRunFactory> = Box::new(
        move |clock: Rc<dyn Clock>,
              backends: Rc<dyn OfflineGraphBackendFactory>|
              -> Result<DeferredOfflineGraphFuture> {
            let backend = backends.create_backend(OfflineGraphBackendConfig {
                phase: MetricsPhase::Profiling,
                prefill_concurrency: None,
                node_policy,
                node_failure,
                events: Rc::new(DiscardOfflineGraphEvents),
            })?;
            let workload = factory.create(clock, backend)?;
            Ok(Box::pin(async move {
                let workload = workload
                    .execute()
                    .await
                    .map_err(|error| anyhow::anyhow!(error.to_string()))?;
                Ok(OfflineGraphExecution {
                    workload,
                    phases: Vec::new(),
                })
            }))
        },
    );
    run_graph_workload_online_deferred(
        engine_config,
        model,
        segments,
        default_max_tokens,
        metrics_config,
        deferred,
    )
}

/// Deferred multi-phase online twin of [`run_graph_workload_offline_deferred`].
///
/// Identical composition (aiperf's [`GraphWorkload`] + [`DynosimSink`] +
/// shared observer/accumulator), swapping only the DES pump for the wall-clock
/// [`drive_real_with_source`] driver and relaxing byte-exact enforcement.
pub fn run_graph_workload_online_deferred(
    engine_config: OfflineEngineConfig,
    model: String,
    segments: Arc<dyn SegmentStore>,
    default_max_tokens: usize,
    metrics_config: MetricsConfig,
    factory: Box<dyn DeferredOfflineGraphRunFactory>,
) -> Result<OfflineDirectGraphReport> {
    anyhow::ensure!(
        !model.trim().is_empty(),
        "online graph model cannot be empty"
    );
    anyhow::ensure!(
        default_max_tokens > 0,
        "online graph default_max_tokens must be positive"
    );

    let clock: Rc<dyn Clock> = crate::clock::real_clock::RealClock::new();
    let start_ns = clock.now_ns();
    let host = EngineHost::new_real(clock.clone(), &engine_config)?;
    let sink = DynosimSink::new(host.clone(), clock.clone(), model);
    let native_metrics = Rc::new(NativeMetricsObserver::new(
        clock.clone(),
        start_ns,
        metrics_config.clone(),
    ));
    let collector = Rc::new(CollectorObserver::new(false));
    let observer: Rc<dyn RequestObserver> = Rc::new(ObserverTee::new(vec![
        collector.clone(),
        native_metrics.clone(),
    ]));
    let backends: Rc<dyn OfflineGraphBackendFactory> = Rc::new(DynosimGraphBackendFactory {
        backend: sink,
        observer,
        native_metrics: native_metrics.clone(),
        materializer: Rc::new(SegmentItemsMaterializer::new(segments)),
        max_tokens: default_max_tokens,
        next_uuid: Rc::new(Cell::new(1)),
    });
    let future = factory.create(clock.clone(), backends)?;
    let result = Rc::new(RefCell::new(None));
    let result_for_body = result.clone();
    let source: Rc<dyn SimEventSource> = host.clone();
    let wakeup = host.submit_wakeup();
    let outcome = drive_real_with_source(source, wakeup, move |_handle| async move {
        *result_for_body.borrow_mut() = Some(future.await);
    })?;
    anyhow::ensure!(
        !outcome.deadlocked,
        "Dynamo online (wall-clock in-process) direct graph run deadlocked"
    );
    let execution = result
        .borrow_mut()
        .take()
        .context("Dynamo online driver exited without a graph phase result")??;
    let wall_ms = clock.now_ns().saturating_sub(start_ns) as f64 / NS_PER_MS;
    let dynamo = host.take_report_at(wall_ms);
    let performance = collector.finish(wall_ms);
    let (performance, dynamo, parity) =
        finish_shared_metrics_enforcing(performance, dynamo, false)?;
    let collection = native_metrics.finish_with_records();
    let mut accumulator = MetricsAccumulator::with_config(metrics_config);
    for (_uuid, record) in &collection.records {
        accumulator.process_record(record);
    }
    let mut profiling_metrics =
        accumulator.export_results(&ExportContext::phase(MetricsPhase::Profiling));
    let warmup_metrics = collection
        .records
        .iter()
        .any(|(_uuid, record)| record.phase == MetricsPhase::Warmup)
        .then(|| accumulator.export_results(&ExportContext::phase(MetricsPhase::Warmup)));
    inject_dynamo_goodput(&mut profiling_metrics, &dynamo);
    Ok(OfflineDirectGraphReport {
        workload: execution.workload,
        performance,
        native_metrics: profiling_metrics,
        warmup_metrics,
        phases: execution.phases,
        dynamo,
        parity,
    })
}

/// One request specification for the native wall-clock online baseline: the
/// exact source-trace hash-block ids plus the requested output length.
#[derive(Clone, Debug)]
pub struct NativeBaselineRequest {
    /// Source-trace block hash ids, converted to tokens by Dynamo's own
    /// `TurnTrace::synthesize_tokens` (the native format AIPerf also ships).
    pub hash_ids: Vec<i64>,
    /// Requested output tokens for this request.
    pub max_output_tokens: usize,
}

/// Flat comparable summary of a native wall-clock online replay run.
#[derive(Clone, Debug)]
pub struct NativeLiveBaseline {
    /// Requests admitted by the native driver.
    pub num_requests: usize,
    /// Requests that reached a terminal completion.
    pub completed_requests: usize,
    /// Summed input tokens across all requests.
    pub total_input_tokens: usize,
    /// Summed output tokens across all requests.
    pub total_output_tokens: usize,
    /// Mean time-to-first-token in milliseconds.
    pub ttft_mean_ms: f64,
    /// Mean end-to-end request latency in milliseconds.
    pub request_latency_mean_ms: f64,
    /// Mean inter-token latency in milliseconds.
    pub inter_token_latency_mean_ms: f64,
    /// Request throughput in requests per second.
    pub request_throughput_rps: f64,
    /// Output-token throughput in tokens per second.
    pub output_throughput_tok_s: f64,
}

/// Run Dynamo's own wall-clock online concurrency replay driver
/// (`simulate_concurrency_live_requests`) as the apples-to-apples baseline for a
/// `replay_mode=online` product run. Requests are built in the native format:
/// each request's tokens come from Dynamo's `TurnTrace::synthesize_tokens`
/// conversion of the same source-trace `hash_ids` the AIPerf runner ships, so
/// the two drivers feed the engine byte-identical inputs. `engine_args_json` is
/// the same inline `MockEngineArgs` object authored on the runner's
/// `backend.config.engine`, so both sides normalize to identical engine args.
///
/// The returned report measures real wall-clock latency (`Instant::elapsed`),
/// exactly like AIPerf's observer, which is what makes the comparison valid.
pub fn native_live_concurrency_baseline(
    engine_args_json: &str,
    requests: &[NativeBaselineRequest],
    input_length: usize,
    max_in_flight: usize,
    num_workers: usize,
) -> Result<NativeLiveBaseline> {
    let block_size = MockEngineArgs::from_json_str(engine_args_json)
        .context("invalid native baseline engine args")?
        .normalized()?
        .block_size;
    let hash_encoder = DynamoTraceHashEncoder;
    let mut direct = Vec::with_capacity(requests.len());
    for (index, request) in requests.iter().enumerate() {
        let tokens = hash_encoder.encode(&request.hash_ids, block_size, input_length)?;
        direct.push(DirectRequest {
            tokens,
            max_output_tokens: request.max_output_tokens,
            output_token_ids: None,
            uuid: Some(Uuid::from_u128(index as u128 + 1)),
            dp_rank: 0,
            arrival_timestamp_ms: None,
            priority: 0,
            strict_priority: 0,
            policy_class: None,
        });
    }
    let args = MockEngineArgs::from_json_str(engine_args_json)
        .context("invalid native baseline engine args")?;
    let report = simulate_concurrency_live_requests(args, direct, max_in_flight, num_workers)?;
    Ok(NativeLiveBaseline {
        num_requests: report.request_counts.num_requests,
        completed_requests: report.request_counts.completed_requests,
        total_input_tokens: report.request_counts.total_input_tokens,
        total_output_tokens: report.request_counts.total_output_tokens,
        ttft_mean_ms: report.latency.ttft.mean_ms,
        request_latency_mean_ms: report.latency.e2e.mean_ms,
        inter_token_latency_mean_ms: report.latency.itl.distribution.mean_ms,
        request_throughput_rps: report.throughput.request_throughput_rps,
        output_throughput_tok_s: report.throughput.output_throughput_tok_s,
    })
}

fn run_scheduled_offline(
    engine_config: OfflineEngineConfig,
    model: String,
    workload: Rc<dyn Workload>,
    stop: StopConfig,
    enforce_stop: bool,
    policies: ScheduledAncillaryPolicies,
    record_processors: Vec<Rc<dyn TurnRecordProcessor>>,
) -> Result<OfflineScheduledReport> {
    anyhow::ensure!(
        policies.url_selector.is_none(),
        "Dynamo offline mode has one in-process endpoint and does not accept URL selection"
    );

    let clock = Rc::new(SimClock::new());
    let host = EngineHost::new(clock.clone(), &engine_config)?;
    let dispatcher: Rc<dyn TurnDispatcher> = DynosimSink::new(host.clone(), clock.clone(), model);
    let result = Rc::new(RefCell::new(None));
    let result_for_body = result.clone();
    let clock_for_body: Rc<dyn Clock> = clock.clone();
    let start_ns = clock.now_ns();
    let source: Rc<dyn SimEventSource> = host.clone();
    let outcome = drive_sim_with_source(clock, source, move |_handle| async move {
        let report = run_scheduled_workload_with_processors(
            workload,
            clock_for_body,
            start_ns,
            dispatcher,
            stop,
            enforce_stop,
            policies,
            record_processors,
        )
        .await;
        *result_for_body.borrow_mut() = Some(report);
    })?;
    anyhow::ensure!(
        !outcome.deadlocked,
        "Dynamo offline scheduled run deadlocked"
    );
    let mut aiperf = result
        .borrow_mut()
        .take()
        .context("Dynamo offline driver exited without a scheduled result")??;
    let dynamo = host.take_report_at(aiperf.performance.throughput.wall_time_ms);
    let (performance, dynamo, parity) = finish_shared_metrics(aiperf.performance, dynamo)?;
    aiperf.performance = performance.clone();
    inject_dynamo_goodput(&mut aiperf.native_metrics, &dynamo);
    Ok(OfflineScheduledReport {
        aiperf,
        performance,
        phases: Vec::new(),
        auxiliary_phase_reports: Vec::new(),
        dynamo,
        parity,
    })
}

/// One scheduled runtime future driven by the offline engine/Clock pump.
pub type OfflineScheduledFuture =
    Pin<Box<dyn Future<Output = Result<OfflineScheduledExecution>> + 'static>>;

/// Factory seam for scheduled runtimes that need the exact offline clock and
/// dispatcher before constructing adaptive observers or Clock-native ramps.
pub trait OfflineScheduledRunFactory {
    /// Build the workload future after the backend has created its one
    /// authoritative virtual clock and dispatcher.
    fn create(
        self: Box<Self>,
        clock: Rc<dyn Clock>,
        start_ns: i64,
        dispatcher: Rc<dyn TurnDispatcher>,
    ) -> OfflineScheduledFuture;
}

impl<F> OfflineScheduledRunFactory for F
where
    F: FnOnce(Rc<dyn Clock>, i64, Rc<dyn TurnDispatcher>) -> OfflineScheduledFuture,
{
    fn create(
        self: Box<Self>,
        clock: Rc<dyn Clock>,
        start_ns: i64,
        dispatcher: Rc<dyn TurnDispatcher>,
    ) -> OfflineScheduledFuture {
        (*self)(clock, start_ns, dispatcher)
    }
}

/// Post-drain report reducer for an offline scheduled execution.
///
/// Implementations retain only worker-local observer state. The backend calls
/// this after `drive_sim_with_source` returns, so reduction cannot delay a
/// clock wakeup or change Dynamo event ordering.
pub trait OfflineScheduledExecutionFinalizer {
    /// Reduce captured facts into the ordinary scheduled execution result.
    fn finish(self: Box<Self>) -> Result<OfflineScheduledExecution>;
}

impl<F> OfflineScheduledExecutionFinalizer for F
where
    F: FnOnce() -> Result<OfflineScheduledExecution>,
{
    fn finish(self: Box<Self>) -> Result<OfflineScheduledExecution> {
        (*self)()
    }
}

/// Scheduled runtime future that stops at the deterministic drain boundary.
pub type DeferredOfflineScheduledFuture =
    Pin<Box<dyn Future<Output = Result<Box<dyn OfflineScheduledExecutionFinalizer>>> + 'static>>;

/// Factory seam for a scheduled runtime with post-`LocalSet` finalization.
pub trait DeferredOfflineScheduledRunFactory {
    /// Build the live workload future after the backend creates its one clock
    /// and dispatcher.
    fn create(
        self: Box<Self>,
        clock: Rc<dyn Clock>,
        start_ns: i64,
        dispatcher: Rc<dyn TurnDispatcher>,
    ) -> DeferredOfflineScheduledFuture;
}

impl<F> DeferredOfflineScheduledRunFactory for F
where
    F: FnOnce(Rc<dyn Clock>, i64, Rc<dyn TurnDispatcher>) -> DeferredOfflineScheduledFuture,
{
    fn create(
        self: Box<Self>,
        clock: Rc<dyn Clock>,
        start_ns: i64,
        dispatcher: Rc<dyn TurnDispatcher>,
    ) -> DeferredOfflineScheduledFuture {
        (*self)(clock, start_ns, dispatcher)
    }
}

/// Drive an arbitrary scheduled runtime over the passive Dynamo engine.
///
/// This is the runner composition seam for scheduled workload factories. It
/// deliberately accepts a trait object rather than a workload-kind enum so a
/// newly linked scheduler can reuse the same engine, clock, parity proof, and
/// report finalization without modifying this module.
pub fn run_scheduled_backend_offline(
    engine_config: OfflineEngineConfig,
    model: String,
    factory: Box<dyn OfflineScheduledRunFactory>,
) -> Result<OfflineScheduledReport> {
    let clock = Rc::new(SimClock::new());
    let host = EngineHost::new(clock.clone(), &engine_config)?;
    let dispatcher: Rc<dyn TurnDispatcher> = DynosimSink::new(host.clone(), clock.clone(), model);
    let start_ns = clock.now_ns();
    let future = factory.create(clock.clone(), start_ns, dispatcher);
    let result = Rc::new(RefCell::new(None));
    let result_for_body = result.clone();
    let source: Rc<dyn SimEventSource> = host.clone();
    let outcome = drive_sim_with_source(clock, source, move |_handle| async move {
        *result_for_body.borrow_mut() = Some(future.await);
    })?;
    anyhow::ensure!(
        !outcome.deadlocked,
        "Dynamo offline scheduled run deadlocked"
    );
    let execution = result
        .borrow_mut()
        .take()
        .context("Dynamo offline driver exited without a scheduled result")??;
    finish_scheduled_backend(host, execution)
}

/// Drive a scheduled runtime against the in-process Dynamo engine under the
/// **wall clock** — the real-time in-process mode (Dynamo's `--replay-mode
/// online`): no sockets, no HTTP, no frontend, but the engine is stepped in real
/// time rather than fast-forwarded. It reuses the exact same `EngineHost`,
/// `DynosimSink`, materializer, observer, and report as
/// [`run_scheduled_backend_offline`]; only the driver differs
/// ([`drive_real_with_source`] vs [`drive_sim_with_source`]) — the
/// `{transport, clock}` seam varying its clock axis.
///
/// Unlike the offline path this is not deterministic (real timers carry jitter),
/// and it advances in real time: trace timing must already be speed-scaled (as
/// the offline `--speedup-ratio` does) or the run takes the trace's real
/// duration. Use it to measure the engine/scheduler/router at its live
/// throughput ceiling with the serving stack removed.
pub fn run_scheduled_backend_online(
    engine_config: OfflineEngineConfig,
    model: String,
    factory: Box<dyn OfflineScheduledRunFactory>,
) -> Result<OfflineScheduledReport> {
    let clock: Rc<dyn Clock> = crate::clock::real_clock::RealClock::new();
    let host = EngineHost::new_real_with_factory_and_delivery(
        clock.clone(),
        &engine_config,
        &NativeDynamoEngineFactory,
        Rc::new(IncrementalOfflineEventDelivery),
    )?;
    let dispatcher: Rc<dyn TurnDispatcher> = DynosimSink::new(host.clone(), clock.clone(), model);
    let start_ns = clock.now_ns();
    let future = factory.create(clock.clone(), start_ns, dispatcher);
    let result = Rc::new(RefCell::new(None));
    let result_for_body = result.clone();
    let source: Rc<dyn SimEventSource> = host.clone();
    let wakeup = host.submit_wakeup();
    let outcome = drive_real_with_source(source, wakeup, move |_handle| async move {
        *result_for_body.borrow_mut() = Some(future.await);
    })?;
    anyhow::ensure!(
        !outcome.deadlocked,
        "Dynamo online (wall-clock in-process) scheduled run deadlocked"
    );
    let execution = result
        .borrow_mut()
        .take()
        .context("Dynamo online driver exited without a scheduled result")??;
    finish_scheduled_backend_online(host, execution)
}

/// Drive a scheduled runtime whose expensive report reduction is deferred
/// until after the Tokio DES runtime has exited.
pub fn run_scheduled_backend_offline_deferred(
    engine_config: OfflineEngineConfig,
    model: String,
    factory: Box<dyn DeferredOfflineScheduledRunFactory>,
) -> Result<OfflineScheduledReport> {
    run_scheduled_backend_offline_deferred_with_delivery(
        engine_config,
        model,
        factory,
        Rc::new(IncrementalOfflineEventDelivery),
    )
}

/// Drive a deferred scheduled runtime with an injected nonterminal event policy.
///
/// This preserves the ordinary scheduled workload and observer path while allowing
/// a phase that has no live TTFT consumer to coalesce Tokio wakeups. Every routed
/// event remains buffered with its original deterministic timestamp and is delivered
/// before the terminal callback.
pub fn run_scheduled_backend_offline_deferred_with_delivery(
    engine_config: OfflineEngineConfig,
    model: String,
    factory: Box<dyn DeferredOfflineScheduledRunFactory>,
    event_delivery: Rc<dyn OfflineEventDeliveryPolicy>,
) -> Result<OfflineScheduledReport> {
    let clock = Rc::new(SimClock::new());
    let host = EngineHost::new_with_delivery(clock.clone(), &engine_config, event_delivery)?;
    let dispatcher: Rc<dyn TurnDispatcher> = DynosimSink::new(host.clone(), clock.clone(), model);
    let start_ns = clock.now_ns();
    let future = factory.create(clock.clone(), start_ns, dispatcher);
    let result = Rc::new(RefCell::new(None));
    let result_for_body = result.clone();
    let source: Rc<dyn SimEventSource> = host.clone();
    let outcome = drive_sim_with_source(clock.clone(), source, move |_handle| async move {
        *result_for_body.borrow_mut() = Some(future.await);
    })?;
    anyhow::ensure!(
        !outcome.deadlocked,
        "Dynamo offline scheduled run deadlocked"
    );
    let finalizer = result
        .borrow_mut()
        .take()
        .context("Dynamo offline driver exited without a scheduled finalizer")??;
    let wall_ms = clock.now_ns().saturating_sub(start_ns) as f64 / 1_000_000.0;
    let dynamo = host.take_report_at(wall_ms);
    let execution = finalizer.finish()?;
    finish_scheduled_backend_with_report(execution, dynamo)
}

/// Deferred online (wall-clock in-process) twin of
/// [`run_scheduled_backend_offline_deferred_with_delivery`].
///
/// Identical scheduled workload, dispatcher, materializer, observer, and report
/// path; only the driver ([`drive_real_with_source`]) and the relaxed parity
/// finalizer differ. Used by the runner's scheduled pair when the authored
/// backend selects `replay_mode = online`.
pub fn run_scheduled_backend_online_deferred_with_delivery(
    engine_config: OfflineEngineConfig,
    model: String,
    factory: Box<dyn DeferredOfflineScheduledRunFactory>,
    event_delivery: Rc<dyn OfflineEventDeliveryPolicy>,
) -> Result<OfflineScheduledReport> {
    let clock: Rc<dyn Clock> = crate::clock::real_clock::RealClock::new();
    let host = EngineHost::new_real_with_factory_and_delivery(
        clock.clone(),
        &engine_config,
        &NativeDynamoEngineFactory,
        event_delivery,
    )?;
    let dispatcher: Rc<dyn TurnDispatcher> = DynosimSink::new(host.clone(), clock.clone(), model);
    let start_ns = clock.now_ns();
    let future = factory.create(clock.clone(), start_ns, dispatcher);
    let result = Rc::new(RefCell::new(None));
    let result_for_body = result.clone();
    let source: Rc<dyn SimEventSource> = host.clone();
    let wakeup = host.submit_wakeup();
    let outcome = drive_real_with_source(source, wakeup, move |_handle| async move {
        *result_for_body.borrow_mut() = Some(future.await);
    })?;
    anyhow::ensure!(
        !outcome.deadlocked,
        "Dynamo online (wall-clock in-process) scheduled run deadlocked"
    );
    let finalizer = result
        .borrow_mut()
        .take()
        .context("Dynamo online driver exited without a scheduled finalizer")??;
    let wall_ms = clock.now_ns().saturating_sub(start_ns) as f64 / 1_000_000.0;
    let dynamo = host.take_report_at(wall_ms);
    let execution = finalizer.finish()?;
    finish_scheduled_backend_with_report_enforcing(execution, dynamo, false)
}

fn finish_scheduled_backend(
    host: Rc<EngineHost>,
    execution: OfflineScheduledExecution,
) -> Result<OfflineScheduledReport> {
    let dynamo = host.take_report_at(execution.performance.throughput.wall_time_ms);
    finish_scheduled_backend_with_report(execution, dynamo)
}

/// Finalize a wall-clock (online) scheduled run: identical to
/// [`finish_scheduled_backend`] but without the byte-exact parity proof, which
/// real-timer latencies cannot satisfy against the engine's internal report.
fn finish_scheduled_backend_online(
    host: Rc<EngineHost>,
    execution: OfflineScheduledExecution,
) -> Result<OfflineScheduledReport> {
    let dynamo = host.take_report_at(execution.performance.throughput.wall_time_ms);
    finish_scheduled_backend_with_report_enforcing(execution, dynamo, false)
}

fn finish_scheduled_backend_with_report(
    execution: OfflineScheduledExecution,
    dynamo: DynamoSimulationReport,
) -> Result<OfflineScheduledReport> {
    finish_scheduled_backend_with_report_enforcing(execution, dynamo, true)
}

fn finish_scheduled_backend_with_report_enforcing(
    mut execution: OfflineScheduledExecution,
    dynamo: DynamoSimulationReport,
    enforce_byte_parity: bool,
) -> Result<OfflineScheduledReport> {
    let (performance, dynamo, parity) =
        finish_shared_metrics_enforcing(execution.performance, dynamo, enforce_byte_parity)?;
    if execution.aggregate_is_profiling {
        execution.profiling.performance = performance.clone();
        inject_dynamo_goodput(&mut execution.profiling.native_metrics, &dynamo);
    }
    Ok(OfflineScheduledReport {
        aiperf: execution.profiling,
        performance,
        phases: execution.phases,
        auxiliary_phase_reports: execution.auxiliary_phase_reports,
        dynamo,
        parity,
    })
}

fn offline_scheduled_policies(
    ancillary: &AncillaryTimingConfig,
    seed: u64,
) -> Result<ScheduledAncillaryPolicies> {
    Ok(ScheduledAncillaryPolicies {
        cancellation_policy: ancillary.cancellation_policy(seed)?,
        url_selector: None,
        phase: Phase::Profiling,
    })
}

/// Replay an authored absolute schedule against the in-process engine.
pub fn run_fixed_schedule_offline(
    engine_config: OfflineEngineConfig,
    model: String,
    conversations: Box<dyn ConversationSource>,
    config: FixedScheduleConfig,
) -> Result<OfflineScheduledReport> {
    run_fixed_schedule_offline_with_ancillary(
        engine_config,
        model,
        conversations,
        config,
        AncillaryTimingConfig::default(),
        0,
    )
}

/// Run authored fixed-schedule replay with Clock-native cancellation policy.
pub fn run_fixed_schedule_offline_with_ancillary(
    engine_config: OfflineEngineConfig,
    model: String,
    conversations: Box<dyn ConversationSource>,
    config: FixedScheduleConfig,
    ancillary: AncillaryTimingConfig,
    seed: u64,
) -> Result<OfflineScheduledReport> {
    ancillary.validate()?;
    anyhow::ensure!(
        !ancillary.has_ramps(),
        "fixed-schedule replay has authored timestamps and does not accept actuator ramps"
    );
    let schedule_source = Rc::new(DatasetFixedScheduleSource::new(config)?);
    let workload: Rc<dyn Workload> =
        Rc::new(FixedScheduleWorkload::new(conversations, schedule_source)?);
    run_scheduled_offline(
        engine_config,
        model,
        workload,
        StopConfig::default(),
        false,
        offline_scheduled_policies(&ancillary, seed)?,
        Vec::new(),
    )
}

/// Run one authored turn per dataset conversation against the in-process engine.
pub fn run_single_turn_dataset_offline(
    engine_config: OfflineEngineConfig,
    model: String,
    conversations: Box<dyn ConversationSource>,
    concurrency: usize,
    record_processors: Vec<Rc<dyn TurnRecordProcessor>>,
) -> Result<OfflineScheduledReport> {
    let workload: Rc<dyn Workload> =
        Rc::new(SingleTurnDatasetWorkload::new(conversations, concurrency)?);
    run_scheduled_offline(
        engine_config,
        model,
        workload,
        StopConfig::default(),
        false,
        ScheduledAncillaryPolicies::default(),
        record_processors,
    )
}

/// Run user-centric virtual-history/churn scheduling against the in-process engine.
pub fn run_user_centric_offline(
    engine_config: OfflineEngineConfig,
    model: String,
    conversations: Box<dyn ConversationSource>,
    config: UserCentricConfig,
    stop: StopConfig,
) -> Result<OfflineScheduledReport> {
    run_user_centric_offline_with_ancillary(
        engine_config,
        model,
        conversations,
        config,
        stop,
        AncillaryTimingConfig::default(),
        0,
    )
}

/// Run user-centric scheduling with deterministic in-process cancellation.
pub fn run_user_centric_offline_with_ancillary(
    engine_config: OfflineEngineConfig,
    model: String,
    conversations: Box<dyn ConversationSource>,
    config: UserCentricConfig,
    stop: StopConfig,
    ancillary: AncillaryTimingConfig,
    seed: u64,
) -> Result<OfflineScheduledReport> {
    run_user_centric_offline_with_adaptive_and_ancillary(
        engine_config,
        model,
        conversations,
        config,
        stop,
        None,
        ancillary,
        seed,
    )
}

/// Run user-centric scheduling with the same optional adaptive user actuator,
/// phase ramps, cancellation, and metrics pipeline used by real HTTP.
#[allow(clippy::too_many_arguments)]
pub fn run_user_centric_offline_with_adaptive_and_ancillary(
    engine_config: OfflineEngineConfig,
    model: String,
    conversations: Box<dyn ConversationSource>,
    config: UserCentricConfig,
    stop: StopConfig,
    adaptive: Option<AdaptiveRunConfig>,
    ancillary: AncillaryTimingConfig,
    seed: u64,
) -> Result<OfflineScheduledReport> {
    let factory: Box<dyn OfflineScheduledRunFactory> = Box::new(
        move |clock: Rc<dyn Clock>, start_ns, dispatcher: Rc<dyn TurnDispatcher>| {
            Box::pin(async move {
                let report = match adaptive {
                    Some(adaptive) => {
                        run_user_centric_adaptive_with_backend(
                            clock,
                            start_ns,
                            dispatcher,
                            vec!["dynosim".to_string()],
                            conversations,
                            config,
                            stop,
                            adaptive,
                            ancillary,
                            seed,
                        )
                        .await
                    }
                    None => {
                        run_user_centric_with_backend(
                            clock,
                            start_ns,
                            dispatcher,
                            vec!["dynosim".to_string()],
                            conversations,
                            config,
                            stop,
                            ancillary,
                            seed,
                        )
                        .await
                    }
                }?;
                Ok(OfflineScheduledExecution::single(report))
            }) as OfflineScheduledFuture
        },
    );
    run_scheduled_backend_offline(engine_config, model, factory)
}

/// Run continuation-priority multi-turn request-rate scheduling against the
/// in-process engine.
pub fn run_request_rate_offline(
    engine_config: OfflineEngineConfig,
    model: String,
    conversations: Box<dyn ConversationSource>,
    config: RequestRateConfig,
    stop: StopConfig,
) -> Result<OfflineScheduledReport> {
    run_request_rate_offline_with_ancillary(
        engine_config,
        model,
        conversations,
        config,
        stop,
        AncillaryTimingConfig::default(),
    )
}

/// Run continuation-priority request-rate scheduling with offline cancellation.
pub fn run_request_rate_offline_with_ancillary(
    engine_config: OfflineEngineConfig,
    model: String,
    conversations: Box<dyn ConversationSource>,
    config: RequestRateConfig,
    stop: StopConfig,
    ancillary: AncillaryTimingConfig,
) -> Result<OfflineScheduledReport> {
    run_request_rate_offline_with_adaptive_and_ancillary(
        engine_config,
        model,
        conversations,
        config,
        stop,
        None,
        ancillary,
    )
}

/// Run continuation-priority request-rate scheduling with the same optional
/// adaptive actuator and ramp ownership rules used by real HTTP.
#[allow(clippy::too_many_arguments)]
pub fn run_request_rate_offline_with_adaptive_and_ancillary(
    engine_config: OfflineEngineConfig,
    model: String,
    conversations: Box<dyn ConversationSource>,
    config: RequestRateConfig,
    stop: StopConfig,
    adaptive: Option<AdaptiveRunConfig>,
    ancillary: AncillaryTimingConfig,
) -> Result<OfflineScheduledReport> {
    let factory: Box<dyn OfflineScheduledRunFactory> = Box::new(
        move |clock: Rc<dyn Clock>, start_ns, dispatcher: Rc<dyn TurnDispatcher>| {
            Box::pin(async move {
                let report = run_request_rate_with_backend(
                    clock,
                    start_ns,
                    dispatcher,
                    vec!["dynosim".to_string()],
                    conversations,
                    config,
                    stop,
                    adaptive,
                    ancillary,
                )
                .await?;
                Ok(OfflineScheduledExecution::single(report))
            }) as OfflineScheduledFuture
        },
    );
    run_scheduled_backend_offline(engine_config, model, factory)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::RngRoot;

    #[test]
    fn python_json_exponents_have_a_sign_and_two_digits() {
        assert_eq!(
            python_json_number_exponents(
                r#"{"small":1e-9,"large":1E9,"wide":1e-123,"text":"1e-9"}"#,
            ),
            r#"{"small":1e-09,"large":1e+09,"wide":1e-123,"text":"1e-9"}"#,
        );
    }

    #[test]
    fn deferred_scheduled_reduction_runs_after_the_tokio_runtime_exits() {
        let factory: Box<dyn DeferredOfflineScheduledRunFactory> = Box::new(
            |_clock: Rc<dyn Clock>, _start_ns, _dispatcher: Rc<dyn TurnDispatcher>| {
                Box::pin(async move {
                    let finalizer: Box<dyn OfflineScheduledExecutionFinalizer> = Box::new(|| {
                        assert!(tokio::runtime::Handle::try_current().is_err());
                        Err(anyhow::anyhow!("deferred-finalizer-proof"))
                    });
                    Ok(finalizer)
                }) as DeferredOfflineScheduledFuture
            },
        );

        let error = match run_scheduled_backend_offline_deferred(
            OfflineEngineConfig::default(),
            "test-model".to_owned(),
            factory,
        ) {
            Ok(_) => panic!("deferred finalizer unexpectedly succeeded"),
            Err(error) => error,
        };
        assert!(format!("{error:#}").contains("deferred-finalizer-proof"));
    }

    fn assert_metric_parity(
        aiperf: &loadgen_core::collector::TraceSimulationReport,
        dynamo: &DynamoSimulationReport,
        parity: OfflineMetricParity,
    ) {
        assert_eq!(
            aiperf.canonical_shared_metric_bytes().unwrap(),
            dynamo.canonical_shared_metric_bytes().unwrap()
        );
        assert_eq!(parity.shared_fields, 74);
        assert_eq!(parity.independently_accumulated_fields, 69);
        assert_eq!(parity.backend_owned_fields, 5);
        assert_eq!(
            parity.serialized_bytes,
            aiperf.canonical_shared_metric_bytes().unwrap().len()
        );
    }

    #[derive(Default)]
    struct RecordingObserver {
        events: RefCell<Vec<&'static str>>,
        admit_ms: Cell<Option<f64>>,
        tokens: Cell<usize>,
        output_batches: Cell<usize>,
        usage_prompt_tokens: Cell<Option<usize>>,
    }

    impl RequestObserver for RecordingObserver {
        fn on_arrival(
            &self,
            _uuid: Uuid,
            _arrival_ms: f64,
            _input_length: usize,
            _requested_output_length: usize,
        ) {
            self.events.borrow_mut().push("arrival");
        }

        fn on_admit(&self, _uuid: Uuid, admit_ms: f64, _reused_input_tokens: usize) {
            self.admit_ms.set(Some(admit_ms));
            self.events.borrow_mut().push("admit");
        }

        fn on_token(&self, _uuid: Uuid, _at_ms: f64) {
            self.tokens.set(self.tokens.get() + 1);
            self.events.borrow_mut().push("token");
        }

        fn on_output_tokens(&self, uuid: Uuid, at_ms: &[f64]) {
            self.output_batches.set(self.output_batches.get() + 1);
            for &timestamp in at_ms {
                self.on_token(uuid, timestamp);
            }
        }

        fn on_usage(&self, _uuid: Uuid, usage: ObservedUsage) {
            self.usage_prompt_tokens.set(usage.prompt_tokens);
            self.events.borrow_mut().push("usage");
        }

        fn on_terminal(&self, _uuid: Uuid, _status: ReplayTerminalStatus) {
            self.events.borrow_mut().push("terminal");
        }
    }

    fn workload(requests: usize) -> SkeletonWorkload {
        SkeletonWorkload {
            num_requests: requests,
            input_tokens: 16,
            output_tokens: 4,
            turns: 1,
            think_time_ms: None,
        }
    }

    struct RecordingEngineFactory<'a> {
        called: &'a Cell<bool>,
    }

    impl OfflineEngineFactory for RecordingEngineFactory<'_> {
        fn build(&self, config: &OfflineEngineConfig) -> Result<Box<dyn SteppableReplay>> {
            self.called.set(true);
            NativeDynamoEngineFactory.build(config)
        }
    }

    struct FixedRequestEncoder;

    impl OfflineRequestEncoder<Request> for FixedRequestEncoder {
        fn encode(&self, _request: &Request) -> Result<Vec<u32>> {
            Ok(vec![11, 22, 33])
        }
    }

    impl OfflineGraphRequestEncoder for FixedRequestEncoder {
        fn encode_graph_messages(&self, _wires: &[Bytes], requested: usize) -> Result<Vec<u32>> {
            Ok(vec![7; requested])
        }
    }

    struct RejectingRequestEncoder;

    impl OfflineRequestEncoder<Request> for RejectingRequestEncoder {
        fn encode(&self, _request: &Request) -> Result<Vec<u32>> {
            panic!("raw token IDs must bypass the offline request encoder")
        }
    }

    impl OfflineGraphRequestEncoder for RejectingRequestEncoder {
        fn encode_graph_messages(&self, _wires: &[Bytes], _requested: usize) -> Result<Vec<u32>> {
            panic!("raw token IDs must bypass the offline graph encoder")
        }
    }

    #[test]
    fn authored_hash_ids_use_dynamos_trace_native_prompt_conversion() {
        let tokens = DynamoTraceHashEncoder.encode(&[11, 12], 4, 6).unwrap();
        assert_eq!(tokens, vec![11, 11, 11, 11, 12, 12]);
        assert!(DynamoTraceHashEncoder.encode(&[-1], 4, 1).is_err());
        assert!(DynamoTraceHashEncoder.encode(&[11], 4, 5).is_err());
    }

    #[test]
    fn local_signal_preserves_wakes_around_registration() {
        let signal = LocalSignal::new();
        let mut before_poll = Box::pin(signal.notified());
        signal.notify();
        let mut context = TaskContext::from_waker(Waker::noop());
        assert!(matches!(
            before_poll.as_mut().poll(&mut context),
            Poll::Ready(())
        ));

        let mut after_poll = Box::pin(signal.notified());
        assert!(matches!(
            after_poll.as_mut().poll(&mut context),
            Poll::Pending
        ));
        signal.notify();
        assert!(matches!(
            after_poll.as_mut().poll(&mut context),
            Poll::Ready(())
        ));
    }

    #[test]
    fn engine_and_request_encoding_are_injected_behind_traits() {
        let clock = Rc::new(SimClock::new());
        let called = Cell::new(false);
        let factory = RecordingEngineFactory { called: &called };
        let host =
            EngineHost::new_with_factory(clock.clone(), &OfflineEngineConfig::default(), &factory)
                .unwrap();
        assert!(called.get());

        let encoder = Rc::new(FixedRequestEncoder);
        let sink = DynosimSink::new_with_encoders(
            host.clone(),
            clock.clone(),
            "model".to_string(),
            encoder.clone(),
            encoder,
        );
        let observer = Rc::new(RecordingObserver::default());
        let observer_for_body = observer.clone();
        let source: Rc<dyn SimEventSource> = host;
        let mut request = workload(1).make_request();
        request.uuid = Uuid::from_u128(123);

        let outcome = drive_sim_with_source(clock, source, move |_handle| async move {
            sink.dispatch(request, observer_for_body.as_ref())
                .await
                .unwrap();
        })
        .unwrap();
        assert!(!outcome.deadlocked);
        assert_eq!(observer.usage_prompt_tokens.get(), Some(3));
    }

    #[test]
    fn authored_raw_token_ids_bypass_all_offline_encoders() {
        use crate::dataset::{
            ComposeConfig, DatasetSource, LoadConfig, LoaderRegistry, TiktokenTokenizer,
            TraceHashAwareRequestMaterializer,
        };
        use crate::endpoints::{
            EndpointId, EndpointRegistry, PreparedEndpointTable, RawEndpointConfig,
        };

        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let local = tokio::task::LocalSet::new();
        let mut compose = ComposeConfig::new("model", RngRoot::new(Some(17)));
        compose.requires_raw_token_ids = true;
        let dataset = local
            .block_on(&runtime, async {
                LoaderRegistry::with_builtin_formats()
                    .unwrap()
                    .build_dataset(
                        Some("single_turn"),
                        &LoadConfig::new(DatasetSource::Inline(serde_json::json!([{
                            "raw_token_ids": [11, 22, 33],
                            "output_length": 2
                        }]))),
                        &compose,
                        &TiktokenTokenizer::builtin(),
                    )
                    .await
            })
            .unwrap();
        let endpoints = EndpointRegistry::builtin().unwrap();
        let endpoint_id = EndpointId::new("vllm_generate").unwrap();
        let endpoint = endpoints
            .prepare(&endpoint_id, RawEndpointConfig::default())
            .unwrap();
        dataset
            .validate_for_endpoint(endpoint.descriptor())
            .unwrap();
        let mut table = PreparedEndpointTable::new();
        let key = table.push(endpoint).unwrap();
        let mut source =
            crate::multiturn::NativeDatasetConversationSource::sequential_with_prepared_endpoint(
                dataset,
                "model",
                2,
                Rc::new(table),
                crate::multiturn::PreparedEndpointReference { key, endpoint_id },
            )
            .unwrap()
            .with_request_materializer(Arc::new(TraceHashAwareRequestMaterializer));
        let turn = source.next(None).unwrap().build_first_turn(None).unwrap();
        assert_eq!(
            turn.raw_token_ids.as_ref().unwrap().resolve().unwrap(),
            &[11, 22, 33]
        );
        assert!(turn.request_body.as_ref().is_some_and(Bytes::is_empty));

        let clock = Rc::new(SimClock::new());
        let host = EngineHost::new(clock.clone(), &OfflineEngineConfig::default()).unwrap();
        let encoder = Rc::new(RejectingRequestEncoder);
        let sink = DynosimSink::new_with_encoders(
            host.clone(),
            clock.clone(),
            "model".into(),
            encoder.clone(),
            encoder,
        );
        let observer = Rc::new(RecordingObserver::default());
        let observer_for_body = observer.clone();
        let source: Rc<dyn SimEventSource> = host;
        let outcome = drive_sim_with_source(clock, source, move |_handle| async move {
            sink.dispatch_turn(turn, observer_for_body.as_ref(), &|_| {})
                .await
                .unwrap();
        })
        .unwrap();
        assert!(!outcome.deadlocked);
        assert_eq!(observer.usage_prompt_tokens.get(), Some(3));
    }

    #[test]
    fn terminal_delivery_coalesces_wakes_without_coalescing_observer_events() {
        let clock = Rc::new(SimClock::new());
        let host = EngineHost::new_with_delivery(
            clock.clone(),
            &OfflineEngineConfig::default(),
            Rc::new(TerminalOfflineEventDelivery),
        )
        .unwrap();
        let encoder = Rc::new(FixedRequestEncoder);
        let sink = DynosimSink::new_with_encoders(
            host.clone(),
            clock.clone(),
            "model".to_string(),
            encoder.clone(),
            encoder,
        );
        let observer = Rc::new(RecordingObserver::default());
        let observer_for_body = observer.clone();
        let source: Rc<dyn SimEventSource> = host;
        let mut request = workload(1).make_request();
        request.uuid = Uuid::from_u128(456);

        let outcome = drive_sim_with_source(clock, source, move |_handle| async move {
            sink.dispatch(request, observer_for_body.as_ref())
                .await
                .unwrap();
        })
        .unwrap();
        assert!(!outcome.deadlocked);
        assert_eq!(observer.tokens.get(), 4);
        assert_eq!(observer.output_batches.get(), 1);
        assert_eq!(observer.usage_prompt_tokens.get(), Some(3));
        let events = observer.events.borrow();
        assert_eq!(events.first(), Some(&"admit"));
        assert_eq!(&events[events.len() - 2..], &["usage", "terminal"]);
    }

    #[test]
    fn closed_loop_runs_end_to_end_without_http() {
        let report = run_paced_offline(
            OfflineEngineConfig::default(),
            "model".to_string(),
            workload(8),
            ArrivalPattern::ConcurrencyBurst,
            None,
            None,
            Some(4),
            None,
            StopConfig {
                total_expected_requests: Some(8),
                ..StopConfig::default()
            },
            7,
            None,
            AncillaryTimingConfig::default(),
        )
        .unwrap();

        assert_eq!(report.aiperf.performance.request_counts.num_requests, 8);
        assert_eq!(
            report.aiperf.performance.request_counts.completed_requests,
            8
        );
        assert_eq!(report.dynamo.request_counts.completed_requests, 8);
        assert!(report.aiperf.performance.latency.ttft.mean_ms > 0.0);
        assert_metric_parity(&report.aiperf.performance, &report.dynamo, report.parity);

        let mut one_ulp_off = report.aiperf.performance.clone();
        one_ulp_off.latency.ttft.mean_ms =
            f64::from_bits(one_ulp_off.latency.ttft.mean_ms.to_bits() + 1);
        let error = match finish_shared_metrics(one_ulp_off, report.dynamo.clone()) {
            Ok(_) => panic!("one-ULP metric drift passed byte-exact parity"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("mean_ttft_ms"));
    }

    /// Fixed native-format token encoder: every AIPerf request dispatches the
    /// exact same token vector that the native `DirectRequest`s carry, so both
    /// drivers feed the engine byte-identical inputs (identical prefix-cache
    /// blocks), not merely equal token counts.
    struct FixedTokensEncoder {
        tokens: Vec<u32>,
    }

    impl OfflineRequestEncoder<Request> for FixedTokensEncoder {
        fn encode(&self, _request: &Request) -> Result<Vec<u32>> {
            Ok(self.tokens.clone())
        }
    }

    impl OfflineGraphRequestEncoder for FixedTokensEncoder {
        fn encode_graph_messages(&self, _wires: &[Bytes], _requested: usize) -> Result<Vec<u32>> {
            Ok(self.tokens.clone())
        }
    }

    /// The `single_pass_engine` path (Dynamo's own `execute_pass` single-runtime)
    /// makes AIPerf's offline concurrency replay **byte-exact** with Dynamo's
    /// native offline driver even under prefill saturation — the regime where the
    /// `AggRuntime` dynamic-admission path defers a re-admitted request's prefill
    /// by one step and diverges. Engine args force saturation (small
    /// `max_num_batched_tokens`, prefix caching off so every request prefills in
    /// full); both drivers consume byte-identical tokens.
    #[test]
    fn offline_single_pass_engine_is_byte_exact_with_native_under_saturation() {
        use dynamo_mocker::replay::simulate_concurrency_requests;
        const N: usize = 64;
        const ISL: usize = 16;
        const OSL: usize = 4;
        const C: usize = 8;
        let eng = r#"{"block_size":16,"enable_prefix_caching":false,"max_num_batched_tokens":32}"#;
        let tokens = vec![7u32; ISL];

        let config = OfflineEngineConfig {
            extra_engine_args: Some(eng.to_string()),
            single_pass_engine: true,
            ..OfflineEngineConfig::default()
        };
        let aiperf = {
            let clock = Rc::new(SimClock::new());
            let host = EngineHost::new(clock.clone(), &config).unwrap();
            let encoder = Rc::new(FixedTokensEncoder {
                tokens: tokens.clone(),
            });
            let sink: Rc<dyn HttpRequestDispatcher> = DynosimSink::new_with_encoders(
                host.clone(),
                clock.clone(),
                "m".to_string(),
                encoder.clone(),
                encoder,
            );
            let result = Rc::new(RefCell::new(None));
            let result_for_body = result.clone();
            let clock_for_body: Rc<dyn Clock> = clock.clone();
            let start_ns = clock.now_ns();
            let source: Rc<dyn SimEventSource> = host.clone();
            let outcome = drive_sim_with_source(clock.clone(), source, move |_h| async move {
                let run = run_paced_with_backend(
                    clock_for_body,
                    start_ns,
                    sink,
                    vec!["dynosim".to_string()],
                    SkeletonWorkload {
                        num_requests: N,
                        input_tokens: ISL,
                        output_tokens: OSL,
                        turns: 1,
                        think_time_ms: None,
                    },
                    ArrivalPattern::ConcurrencyBurst,
                    None,
                    None,
                    Some(C),
                    None,
                    StopConfig {
                        total_expected_requests: Some(N as u64),
                        ..StopConfig::default()
                    },
                    7,
                    None,
                    AncillaryTimingConfig::default(),
                )
                .await;
                *result_for_body.borrow_mut() = Some(run);
            })
            .unwrap();
            assert!(!outcome.deadlocked);
            let run = result.borrow_mut().take().unwrap().unwrap();
            host.take_report_at(run.performance.throughput.wall_time_ms)
        };

        let requests: Vec<_> = (0..N)
            .map(|i| DirectRequest {
                tokens: tokens.clone(),
                max_output_tokens: OSL,
                output_token_ids: None,
                uuid: Some(Uuid::from_u128(i as u128 + 1)),
                dp_rank: 0,
                arrival_timestamp_ms: None,
                priority: 0,
                strict_priority: 0,
                policy_class: None,
            })
            .collect();
        let native = simulate_concurrency_requests(
            MockEngineArgs::from_json_str(eng).unwrap(),
            requests,
            C,
            1,
        )
        .unwrap();

        assert_eq!(
            aiperf.request_counts.completed_requests,
            native.request_counts.completed_requests
        );
        assert_eq!(
            aiperf.request_counts.total_output_tokens,
            native.request_counts.total_output_tokens
        );
        // Byte-exact latency under saturation — the property that failed with the
        // AggRuntime dynamic-admission path. Equal to within a few ULP: the only
        // residual is floating-point accumulation order between the two
        // collectors, not a scheduling divergence (which was tens of percent).
        let within_1_ulp = |name: &str, a: f64, n: f64| {
            let tol = n.abs() * 8.0 * f64::EPSILON;
            assert!(
                (a - n).abs() <= tol,
                "{name} not byte-exact: aiperf={a} native={n} delta={}",
                (a - n).abs()
            );
        };
        within_1_ulp(
            "ttft",
            aiperf.latency.ttft.mean_ms,
            native.latency.ttft.mean_ms,
        );
        within_1_ulp(
            "e2e",
            aiperf.latency.e2e.mean_ms,
            native.latency.e2e.mean_ms,
        );
        within_1_ulp(
            "itl",
            aiperf.latency.itl.distribution.mean_ms,
            native.latency.itl.distribution.mean_ms,
        );
    }

    /// The multi-worker `SteppableAgg` PUSH path is byte-exact with Dynamo's own
    /// native offline `AggRuntime::run()` (pull) driver under prefill saturation —
    /// the regime where AIPerf's dynamic `SlotPool` admission previously deferred a
    /// re-admitted request's prefill by one drain cycle and diverged tens of
    /// percent. The mocker's steppable delta-cycle (EVALUATE completions → caller
    /// submits the replacement at the completion instant → COMMIT drives it now)
    /// closes that gap for round-robin aggregated replay with more than one worker,
    /// where `single_pass_engine` does not apply. Same engine args, same tokens.
    #[test]
    fn offline_agg_deferred_drive_is_byte_exact_with_native_multi_worker() {
        use dynamo_mocker::replay::simulate_concurrency_requests;
        const N: usize = 96;
        const ISL: usize = 16;
        const OSL: usize = 4;
        const C: usize = 12;
        const WORKERS: usize = 2;
        let eng = r#"{"block_size":16,"enable_prefix_caching":false,"max_num_batched_tokens":32}"#;
        let tokens = vec![7u32; ISL];

        let config = OfflineEngineConfig {
            extra_engine_args: Some(eng.to_string()),
            single_pass_engine: false,
            topology: OfflineTopology::Aggregated,
            workers: WORKERS,
            ..OfflineEngineConfig::default()
        };
        let aiperf = {
            let clock = Rc::new(SimClock::new());
            let host = EngineHost::new(clock.clone(), &config).unwrap();
            let encoder = Rc::new(FixedTokensEncoder {
                tokens: tokens.clone(),
            });
            let sink: Rc<dyn HttpRequestDispatcher> = DynosimSink::new_with_encoders(
                host.clone(),
                clock.clone(),
                "m".to_string(),
                encoder.clone(),
                encoder,
            );
            let result = Rc::new(RefCell::new(None));
            let result_for_body = result.clone();
            let clock_for_body: Rc<dyn Clock> = clock.clone();
            let start_ns = clock.now_ns();
            let source: Rc<dyn SimEventSource> = host.clone();
            let outcome = drive_sim_with_source(clock.clone(), source, move |_h| async move {
                let run = run_paced_with_backend(
                    clock_for_body,
                    start_ns,
                    sink,
                    vec!["dynosim".to_string()],
                    SkeletonWorkload {
                        num_requests: N,
                        input_tokens: ISL,
                        output_tokens: OSL,
                        turns: 1,
                        think_time_ms: None,
                    },
                    ArrivalPattern::ConcurrencyBurst,
                    None,
                    None,
                    Some(C),
                    None,
                    StopConfig {
                        total_expected_requests: Some(N as u64),
                        ..StopConfig::default()
                    },
                    7,
                    None,
                    AncillaryTimingConfig::default(),
                )
                .await;
                *result_for_body.borrow_mut() = Some(run);
            })
            .unwrap();
            assert!(!outcome.deadlocked);
            let run = result.borrow_mut().take().unwrap().unwrap();
            host.take_report_at(run.performance.throughput.wall_time_ms)
        };

        let requests: Vec<_> = (0..N)
            .map(|i| DirectRequest {
                tokens: tokens.clone(),
                max_output_tokens: OSL,
                output_token_ids: None,
                uuid: Some(Uuid::from_u128(i as u128 + 1)),
                dp_rank: 0,
                arrival_timestamp_ms: None,
                priority: 0,
                strict_priority: 0,
                policy_class: None,
            })
            .collect();
        let native = simulate_concurrency_requests(
            MockEngineArgs::from_json_str(eng).unwrap(),
            requests,
            C,
            WORKERS,
        )
        .unwrap();

        assert_eq!(
            aiperf.request_counts.completed_requests,
            native.request_counts.completed_requests
        );
        assert_eq!(
            aiperf.request_counts.total_output_tokens,
            native.request_counts.total_output_tokens
        );
        let within_1_ulp = |name: &str, a: f64, n: f64| {
            let tol = n.abs() * 8.0 * f64::EPSILON;
            assert!(
                (a - n).abs() <= tol,
                "{name} not byte-exact: aiperf={a} native={n} delta={}",
                (a - n).abs()
            );
        };
        within_1_ulp(
            "ttft",
            aiperf.latency.ttft.mean_ms,
            native.latency.ttft.mean_ms,
        );
        within_1_ulp(
            "e2e",
            aiperf.latency.e2e.mean_ms,
            native.latency.e2e.mean_ms,
        );
        within_1_ulp(
            "itl",
            aiperf.latency.itl.distribution.mean_ms,
            native.latency.itl.distribution.mean_ms,
        );
    }

    /// The KV-router `SteppableAgg` PUSH path is byte-exact with Dynamo's own
    /// native offline KV-router `run()` (pull) driver under prefill saturation.
    /// The router admits its own queued requests on completion, so the delta-cycle
    /// still applies: the EVALUATE yield lets AIPerf's `SlotPool` submit the
    /// replacement (which the router queues and admits at the completion instant)
    /// before the freed worker is driven.
    #[test]
    fn offline_kv_router_deferred_drive_is_byte_exact_with_native_multi_worker() {
        use dynamo_mocker::replay::simulate_concurrency_requests_with_router_mode;
        const N: usize = 96;
        const ISL: usize = 16;
        const OSL: usize = 4;
        const C: usize = 12;
        const WORKERS: usize = 2;
        let eng = r#"{"block_size":16,"enable_prefix_caching":false,"max_num_batched_tokens":32}"#;
        let tokens = vec![7u32; ISL];
        let router_json = "{}";

        let config = OfflineEngineConfig {
            extra_engine_args: Some(eng.to_string()),
            single_pass_engine: false,
            topology: OfflineTopology::Aggregated,
            workers: WORKERS,
            router_mode: OfflineRouterMode::Kv,
            router_config: Some(router_json.to_string()),
            ..OfflineEngineConfig::default()
        };
        let aiperf = {
            let clock = Rc::new(SimClock::new());
            let host = EngineHost::new(clock.clone(), &config).unwrap();
            let encoder = Rc::new(FixedTokensEncoder {
                tokens: tokens.clone(),
            });
            let sink: Rc<dyn HttpRequestDispatcher> = DynosimSink::new_with_encoders(
                host.clone(),
                clock.clone(),
                "m".to_string(),
                encoder.clone(),
                encoder,
            );
            let result = Rc::new(RefCell::new(None));
            let result_for_body = result.clone();
            let clock_for_body: Rc<dyn Clock> = clock.clone();
            let start_ns = clock.now_ns();
            let source: Rc<dyn SimEventSource> = host.clone();
            let outcome = drive_sim_with_source(clock.clone(), source, move |_h| async move {
                let run = run_paced_with_backend(
                    clock_for_body,
                    start_ns,
                    sink,
                    vec!["dynosim".to_string()],
                    SkeletonWorkload {
                        num_requests: N,
                        input_tokens: ISL,
                        output_tokens: OSL,
                        turns: 1,
                        think_time_ms: None,
                    },
                    ArrivalPattern::ConcurrencyBurst,
                    None,
                    None,
                    Some(C),
                    None,
                    StopConfig {
                        total_expected_requests: Some(N as u64),
                        ..StopConfig::default()
                    },
                    7,
                    None,
                    AncillaryTimingConfig::default(),
                )
                .await;
                *result_for_body.borrow_mut() = Some(run);
            })
            .unwrap();
            assert!(!outcome.deadlocked);
            let run = result.borrow_mut().take().unwrap().unwrap();
            host.take_report_at(run.performance.throughput.wall_time_ms)
        };

        let requests: Vec<_> = (0..N)
            .map(|i| DirectRequest {
                tokens: tokens.clone(),
                max_output_tokens: OSL,
                output_token_ids: None,
                uuid: Some(Uuid::from_u128(i as u128 + 1)),
                dp_rank: 0,
                arrival_timestamp_ms: None,
                priority: 0,
                strict_priority: 0,
                policy_class: None,
            })
            .collect();
        let router_config: ReplayKvRouterConfig = serde_json::from_str(router_json).unwrap();
        let native = simulate_concurrency_requests_with_router_mode(
            MockEngineArgs::from_json_str(eng).unwrap(),
            Some(router_config),
            None,
            requests,
            C,
            WORKERS,
            ReplayRouterMode::KvRouter,
            DynamoSlaThresholds::default(),
        )
        .unwrap();

        assert_eq!(
            aiperf.request_counts.completed_requests,
            native.request_counts.completed_requests
        );
        assert_eq!(
            aiperf.request_counts.total_output_tokens,
            native.request_counts.total_output_tokens
        );
        let within_1_ulp = |name: &str, a: f64, n: f64| {
            let tol = n.abs() * 8.0 * f64::EPSILON;
            assert!(
                (a - n).abs() <= tol,
                "{name} not byte-exact: aiperf={a} native={n} delta={}",
                (a - n).abs()
            );
        };
        within_1_ulp(
            "ttft",
            aiperf.latency.ttft.mean_ms,
            native.latency.ttft.mean_ms,
        );
        within_1_ulp(
            "e2e",
            aiperf.latency.e2e.mean_ms,
            native.latency.e2e.mean_ms,
        );
        within_1_ulp(
            "itl",
            aiperf.latency.itl.distribution.mean_ms,
            native.latency.itl.distribution.mean_ms,
        );
    }

    /// The disaggregated `SteppableDisagg` PUSH path is byte-exact with Dynamo's
    /// own native offline disaggregated `run()` (pull) driver under prefill
    /// saturation. The steppable delta-cycle yields on the terminal decode
    /// completion so AIPerf's `SlotPool` admits the replacement at that instant
    /// and its prefill is driven now on the prefill pool — closing the same
    /// one-drain-cycle deferral the aggregated path had.
    #[test]
    fn offline_disagg_deferred_drive_is_byte_exact_with_native() {
        use dynamo_mocker::replay::{
            OfflineDisaggReplayConfig, simulate_concurrency_requests_disagg_with_router_mode,
        };
        const N: usize = 80;
        const ISL: usize = 16;
        const OSL: usize = 4;
        const C: usize = 10;
        let eng = r#"{"block_size":16,"enable_prefix_caching":false,"max_num_batched_tokens":32}"#;
        let tokens = vec![7u32; ISL];

        let config = OfflineEngineConfig {
            extra_engine_args: Some(eng.to_string()),
            single_pass_engine: false,
            topology: OfflineTopology::Disaggregated,
            prefill_workers: 1,
            decode_workers: 1,
            ..OfflineEngineConfig::default()
        };
        let aiperf = {
            let clock = Rc::new(SimClock::new());
            let host = EngineHost::new(clock.clone(), &config).unwrap();
            let encoder = Rc::new(FixedTokensEncoder {
                tokens: tokens.clone(),
            });
            let sink: Rc<dyn HttpRequestDispatcher> = DynosimSink::new_with_encoders(
                host.clone(),
                clock.clone(),
                "m".to_string(),
                encoder.clone(),
                encoder,
            );
            let result = Rc::new(RefCell::new(None));
            let result_for_body = result.clone();
            let clock_for_body: Rc<dyn Clock> = clock.clone();
            let start_ns = clock.now_ns();
            let source: Rc<dyn SimEventSource> = host.clone();
            let outcome = drive_sim_with_source(clock.clone(), source, move |_h| async move {
                let run = run_paced_with_backend(
                    clock_for_body,
                    start_ns,
                    sink,
                    vec!["dynosim".to_string()],
                    SkeletonWorkload {
                        num_requests: N,
                        input_tokens: ISL,
                        output_tokens: OSL,
                        turns: 1,
                        think_time_ms: None,
                    },
                    ArrivalPattern::ConcurrencyBurst,
                    None,
                    None,
                    Some(C),
                    None,
                    StopConfig {
                        total_expected_requests: Some(N as u64),
                        ..StopConfig::default()
                    },
                    7,
                    None,
                    AncillaryTimingConfig::default(),
                )
                .await;
                *result_for_body.borrow_mut() = Some(run);
            })
            .unwrap();
            assert!(!outcome.deadlocked);
            let run = result.borrow_mut().take().unwrap().unwrap();
            host.take_report_at(run.performance.throughput.wall_time_ms)
        };

        let requests: Vec<_> = (0..N)
            .map(|i| DirectRequest {
                tokens: tokens.clone(),
                max_output_tokens: OSL,
                output_token_ids: None,
                uuid: Some(Uuid::from_u128(i as u128 + 1)),
                dp_rank: 0,
                arrival_timestamp_ms: None,
                priority: 0,
                strict_priority: 0,
                policy_class: None,
            })
            .collect();
        let mut prefill_args = MockEngineArgs::from_json_str(eng).unwrap();
        prefill_args.worker_type = WorkerType::Prefill;
        let mut decode_args = MockEngineArgs::from_json_str(eng).unwrap();
        decode_args.worker_type = WorkerType::Decode;
        let native = simulate_concurrency_requests_disagg_with_router_mode(
            OfflineDisaggReplayConfig {
                prefill_args,
                decode_args,
                num_prefill_workers: 1,
                num_decode_workers: 1,
            },
            None,
            None,
            requests,
            C,
            ReplayRouterMode::RoundRobin,
            DynamoSlaThresholds::default(),
        )
        .unwrap();

        assert_eq!(
            aiperf.request_counts.completed_requests,
            native.request_counts.completed_requests
        );
        assert_eq!(
            aiperf.request_counts.total_output_tokens,
            native.request_counts.total_output_tokens
        );
        let within_1_ulp = |name: &str, a: f64, n: f64| {
            let tol = n.abs() * 8.0 * f64::EPSILON;
            assert!(
                (a - n).abs() <= tol,
                "{name} not byte-exact: aiperf={a} native={n} delta={}",
                (a - n).abs()
            );
        };
        within_1_ulp(
            "ttft",
            aiperf.latency.ttft.mean_ms,
            native.latency.ttft.mean_ms,
        );
        within_1_ulp(
            "e2e",
            aiperf.latency.e2e.mean_ms,
            native.latency.e2e.mean_ms,
        );
        within_1_ulp(
            "itl",
            aiperf.latency.itl.distribution.mean_ms,
            native.latency.itl.distribution.mean_ms,
        );
    }

    /// The disaggregated KV-router `SteppableDisagg` PUSH path is byte-exact with
    /// Dynamo's own native offline disaggregated KV-router `run()` (pull) driver
    /// under prefill saturation — the last of the four offline routing/topology
    /// combinations, completing byte-exactness for every offline engine/mode.
    #[test]
    fn offline_disagg_kv_router_deferred_drive_is_byte_exact_with_native() {
        use dynamo_mocker::replay::{
            OfflineDisaggReplayConfig, simulate_concurrency_requests_disagg_with_router_mode,
        };
        const N: usize = 80;
        const ISL: usize = 16;
        const OSL: usize = 4;
        const C: usize = 10;
        let eng = r#"{"block_size":16,"enable_prefix_caching":false,"max_num_batched_tokens":32}"#;
        let tokens = vec![7u32; ISL];
        let router_json = "{}";

        let config = OfflineEngineConfig {
            extra_engine_args: Some(eng.to_string()),
            single_pass_engine: false,
            topology: OfflineTopology::Disaggregated,
            prefill_workers: 1,
            decode_workers: 1,
            router_mode: OfflineRouterMode::Kv,
            router_config: Some(router_json.to_string()),
            ..OfflineEngineConfig::default()
        };
        let aiperf = {
            let clock = Rc::new(SimClock::new());
            let host = EngineHost::new(clock.clone(), &config).unwrap();
            let encoder = Rc::new(FixedTokensEncoder {
                tokens: tokens.clone(),
            });
            let sink: Rc<dyn HttpRequestDispatcher> = DynosimSink::new_with_encoders(
                host.clone(),
                clock.clone(),
                "m".to_string(),
                encoder.clone(),
                encoder,
            );
            let result = Rc::new(RefCell::new(None));
            let result_for_body = result.clone();
            let clock_for_body: Rc<dyn Clock> = clock.clone();
            let start_ns = clock.now_ns();
            let source: Rc<dyn SimEventSource> = host.clone();
            let outcome = drive_sim_with_source(clock.clone(), source, move |_h| async move {
                let run = run_paced_with_backend(
                    clock_for_body,
                    start_ns,
                    sink,
                    vec!["dynosim".to_string()],
                    SkeletonWorkload {
                        num_requests: N,
                        input_tokens: ISL,
                        output_tokens: OSL,
                        turns: 1,
                        think_time_ms: None,
                    },
                    ArrivalPattern::ConcurrencyBurst,
                    None,
                    None,
                    Some(C),
                    None,
                    StopConfig {
                        total_expected_requests: Some(N as u64),
                        ..StopConfig::default()
                    },
                    7,
                    None,
                    AncillaryTimingConfig::default(),
                )
                .await;
                *result_for_body.borrow_mut() = Some(run);
            })
            .unwrap();
            assert!(!outcome.deadlocked);
            let run = result.borrow_mut().take().unwrap().unwrap();
            host.take_report_at(run.performance.throughput.wall_time_ms)
        };

        let requests: Vec<_> = (0..N)
            .map(|i| DirectRequest {
                tokens: tokens.clone(),
                max_output_tokens: OSL,
                output_token_ids: None,
                uuid: Some(Uuid::from_u128(i as u128 + 1)),
                dp_rank: 0,
                arrival_timestamp_ms: None,
                priority: 0,
                strict_priority: 0,
                policy_class: None,
            })
            .collect();
        let mut prefill_args = MockEngineArgs::from_json_str(eng).unwrap();
        prefill_args.worker_type = WorkerType::Prefill;
        let mut decode_args = MockEngineArgs::from_json_str(eng).unwrap();
        decode_args.worker_type = WorkerType::Decode;
        let router_config: ReplayKvRouterConfig = serde_json::from_str(router_json).unwrap();
        let native = simulate_concurrency_requests_disagg_with_router_mode(
            OfflineDisaggReplayConfig {
                prefill_args,
                decode_args,
                num_prefill_workers: 1,
                num_decode_workers: 1,
            },
            Some(router_config),
            None,
            requests,
            C,
            ReplayRouterMode::KvRouter,
            DynamoSlaThresholds::default(),
        )
        .unwrap();

        assert_eq!(
            aiperf.request_counts.completed_requests,
            native.request_counts.completed_requests
        );
        assert_eq!(
            aiperf.request_counts.total_output_tokens,
            native.request_counts.total_output_tokens
        );
        let within_1_ulp = |name: &str, a: f64, n: f64| {
            let tol = n.abs() * 8.0 * f64::EPSILON;
            assert!(
                (a - n).abs() <= tol,
                "{name} not byte-exact: aiperf={a} native={n} delta={}",
                (a - n).abs()
            );
        };
        within_1_ulp(
            "ttft",
            aiperf.latency.ttft.mean_ms,
            native.latency.ttft.mean_ms,
        );
        within_1_ulp(
            "e2e",
            aiperf.latency.e2e.mean_ms,
            native.latency.e2e.mean_ms,
        );
        within_1_ulp(
            "itl",
            aiperf.latency.itl.distribution.mean_ms,
            native.latency.itl.distribution.mean_ms,
        );
    }

    /// Apples-to-apples gate: AIPerf's own online flow versus Dynamo's native
    /// wall-clock online replay driver, both under the real clock, on the same
    /// engine, with byte-identical request tokens produced by Dynamo's own
    /// `TurnTrace::synthesize_tokens` hash-block conversion (the native format).
    /// Dynamo's `simulate_concurrency_live_requests` is the real-clock native
    /// baseline — its report measures wall-clock latency via `Instant::elapsed`,
    /// exactly like AIPerf's observer. The gate: request/token counts are exact,
    /// every latency stat is within 3%, and AIPerf's throughput is at least the
    /// native throughput.
    #[test]
    fn online_matches_native_dynamo_live_replay_apples_to_apples() {
        use dynamo_mocker::replay::simulate_concurrency_live_requests;

        const REQUESTS: usize = 32;
        const OSL: usize = 8;
        // No driver-side admission queue: every request is admitted at t=0 on
        // both sides, so TTFT reflects only engine-side batching (identical on
        // both) rather than a driver queue-wait artifact.
        const CONCURRENCY: usize = REQUESTS;

        // Build the shared inputs in the native format: one prefill block of
        // hash-derived tokens via Dynamo's own conversion. Identical across all
        // requests, so the engine's cache-reuse pattern is identical on both
        // drivers.
        let args = MockEngineArgs::default().normalized().unwrap();
        let block_size = args.block_size;
        let isl = block_size;
        let tokens = DynamoTraceHashEncoder
            .encode(&[7i64], block_size, isl)
            .unwrap();
        assert_eq!(tokens.len(), isl);

        // AIPerf: the normal online flow (paced workload -> SlotPool admission ->
        // DynosimSink -> observer/collector) under the real clock, but with
        // the fixed native-format token encoder so the engine sees exactly the
        // native tokens rather than text-encoded synthetic ones.
        let aiperf = {
            let clock: Rc<dyn Clock> = crate::clock::real_clock::RealClock::new();
            let host = EngineHost::new_real_with_factory_and_delivery(
                clock.clone(),
                &OfflineEngineConfig::default(),
                &NativeDynamoEngineFactory,
                Rc::new(IncrementalOfflineEventDelivery),
            )
            .unwrap();
            let encoder = Rc::new(FixedTokensEncoder {
                tokens: tokens.clone(),
            });
            let sink: Rc<dyn HttpRequestDispatcher> = DynosimSink::new_with_encoders(
                host.clone(),
                clock.clone(),
                "model".to_string(),
                encoder.clone(),
                encoder,
            );
            let result = Rc::new(RefCell::new(None));
            let result_for_body = result.clone();
            let clock_for_body: Rc<dyn Clock> = clock.clone();
            let start_ns = clock.now_ns();
            let source: Rc<dyn SimEventSource> = host.clone();
            let wakeup = host.submit_wakeup();
            let outcome = drive_real_with_source(source, wakeup, move |_handle| async move {
                let run = run_paced_with_backend(
                    clock_for_body,
                    start_ns,
                    sink,
                    vec!["dynosim".to_string()],
                    SkeletonWorkload {
                        num_requests: REQUESTS,
                        input_tokens: isl,
                        output_tokens: OSL,
                        turns: 1,
                        think_time_ms: None,
                    },
                    ArrivalPattern::ConcurrencyBurst,
                    None,
                    None,
                    Some(CONCURRENCY),
                    None,
                    StopConfig {
                        total_expected_requests: Some(REQUESTS as u64),
                        ..StopConfig::default()
                    },
                    7,
                    None,
                    AncillaryTimingConfig::default(),
                )
                .await;
                *result_for_body.borrow_mut() = Some(run);
            })
            .unwrap();
            assert!(!outcome.deadlocked);
            result.borrow_mut().take().unwrap().unwrap().performance
        };

        // Native baseline: Dynamo's own wall-clock online driver, same engine
        // args, byte-identical tokens, same in-flight concurrency, one worker.
        let requests = (0..REQUESTS)
            .map(|index| DirectRequest {
                tokens: tokens.clone(),
                max_output_tokens: OSL,
                output_token_ids: None,
                uuid: Some(Uuid::from_u128(index as u128 + 1)),
                dp_rank: 0,
                arrival_timestamp_ms: None,
                priority: 0,
                strict_priority: 0,
                policy_class: None,
            })
            .collect::<Vec<_>>();
        let native =
            simulate_concurrency_live_requests(MockEngineArgs::default(), requests, CONCURRENCY, 1)
                .unwrap();

        // Counts: exact (0% divergence — the sim results are 100% comparable).
        assert_eq!(
            aiperf.request_counts.num_requests,
            native.request_counts.num_requests
        );
        assert_eq!(
            aiperf.request_counts.completed_requests,
            native.request_counts.completed_requests
        );
        assert_eq!(
            aiperf.request_counts.total_input_tokens,
            native.request_counts.total_input_tokens
        );
        assert_eq!(
            aiperf.request_counts.total_output_tokens,
            native.request_counts.total_output_tokens
        );

        // Latency: every stat within 3% (an absolute 1ms floor keeps sub-ms
        // synthetic latencies from making a hair-trigger ratio flaky; real
        // traces have far larger latencies where the fixed real overhead is a
        // much smaller fraction).
        let within_3pct = |name: &str, a: f64, n: f64| {
            let delta = (a - n).abs();
            let tol = (n.abs() * 0.03).max(1.0);
            assert!(
                delta <= tol,
                "{name}: aiperf={a:.4} native={n:.4} delta={delta:.4} exceeds 3% (tol={tol:.4})"
            );
        };
        within_3pct(
            "ttft.mean",
            aiperf.latency.ttft.mean_ms,
            native.latency.ttft.mean_ms,
        );
        within_3pct(
            "ttft.p90",
            aiperf.latency.ttft.p90_ms,
            native.latency.ttft.p90_ms,
        );
        within_3pct(
            "e2e.mean",
            aiperf.latency.e2e.mean_ms,
            native.latency.e2e.mean_ms,
        );
        within_3pct(
            "e2e.p90",
            aiperf.latency.e2e.p90_ms,
            native.latency.e2e.p90_ms,
        );
        within_3pct(
            "itl.mean",
            aiperf.latency.itl.distribution.mean_ms,
            native.latency.itl.distribution.mean_ms,
        );
        within_3pct(
            "tpot.mean",
            aiperf.latency.tpot.mean_ms,
            native.latency.tpot.mean_ms,
        );

        // Throughput: AIPerf's driver is at least as fast as the native driver.
        // A small slack absorbs real-timer jitter on the (already fast) run.
        let slack = 0.97;
        assert!(
            aiperf.throughput.request_throughput_rps
                >= native.throughput.request_throughput_rps * slack,
            "aiperf rps {:.3} < native rps {:.3}",
            aiperf.throughput.request_throughput_rps,
            native.throughput.request_throughput_rps
        );
        assert!(
            aiperf.throughput.output_throughput_tok_s
                >= native.throughput.output_throughput_tok_s * slack,
            "aiperf out t/s {:.3} < native out t/s {:.3}",
            aiperf.throughput.output_throughput_tok_s,
            native.throughput.output_throughput_tok_s
        );
    }

    #[test]
    fn online_wall_clock_runs_end_to_end_in_process_without_http() {
        // The wall-clock (drive_real_with_source) driver runs the same in-process
        // Dynamo engine as the offline DES path — no sockets, no frontend — but in
        // real time. Request/token counts stay exact; latencies come from real
        // timers so byte-exact parity is (correctly) not asserted.
        let report = run_paced_online(
            OfflineEngineConfig::default(),
            "model".to_string(),
            workload(8),
            ArrivalPattern::ConcurrencyBurst,
            None,
            None,
            Some(4),
            None,
            StopConfig {
                total_expected_requests: Some(8),
                ..StopConfig::default()
            },
            7,
            None,
            AncillaryTimingConfig::default(),
        )
        .unwrap();

        assert_eq!(report.aiperf.performance.request_counts.num_requests, 8);
        assert_eq!(
            report.aiperf.performance.request_counts.completed_requests,
            8
        );
        assert_eq!(report.dynamo.request_counts.completed_requests, 8);
        // Same in-process engine, real clock: it produced real, non-negative
        // per-request timing without ever opening a socket.
        assert!(report.aiperf.performance.latency.ttft.mean_ms >= 0.0);
    }

    #[test]
    fn engine_admission_tokens_usage_and_terminal_flow_through_the_observer() {
        let clock = Rc::new(SimClock::new());
        let host = EngineHost::new(clock.clone(), &OfflineEngineConfig::default()).unwrap();
        let sink = DynosimSink::new(host.clone(), clock.clone(), "model".to_string());
        let observer = Rc::new(RecordingObserver::default());
        let observer_for_body = observer.clone();
        let source: Rc<dyn SimEventSource> = host.clone();
        let mut request = workload(1).make_request();
        request.uuid = Uuid::from_u128(99);
        let uuid = request.uuid;

        let outcome = drive_sim_with_source(clock, source, move |_handle| async move {
            observer_for_body.on_arrival(
                uuid,
                0.0,
                request.input_length,
                request.max_output_tokens,
            );
            sink.dispatch(request, observer_for_body.as_ref())
                .await
                .unwrap();
        })
        .unwrap();

        assert!(!outcome.deadlocked);
        assert_eq!(observer.tokens.get(), 4);
        assert!(observer.admit_ms.get().is_some_and(|value| value >= 0.0));
        let events = observer.events.borrow();
        assert_eq!(events.first(), Some(&"arrival"));
        let admit = events.iter().position(|event| *event == "admit").unwrap();
        let first_token = events.iter().position(|event| *event == "token").unwrap();
        let usage = events.iter().position(|event| *event == "usage").unwrap();
        let terminal = events
            .iter()
            .position(|event| *event == "terminal")
            .unwrap();
        assert!(admit < first_token && first_token < usage && usage < terminal);
        assert_eq!(host.take_report().request_counts.completed_requests, 1);
    }

    #[test]
    fn constant_rate_runs_on_the_shared_virtual_clock() {
        let engine = OfflineEngineConfig {
            topology: OfflineTopology::Aggregated,
            ..OfflineEngineConfig::default()
        };
        let report = run_paced_offline(
            engine,
            "model".to_string(),
            workload(12),
            ArrivalPattern::Constant,
            Some(1_000.0),
            None,
            None,
            None,
            StopConfig {
                total_expected_requests: Some(12),
                ..StopConfig::default()
            },
            11,
            None,
            AncillaryTimingConfig::default(),
        )
        .unwrap();

        assert_eq!(report.aiperf.performance.request_counts.num_requests, 12);
        assert_eq!(
            report.aiperf.performance.request_counts.completed_requests,
            12
        );
        assert!(report.aiperf.performance.throughput.wall_time_ms >= 12.0);
        assert_metric_parity(&report.aiperf.performance, &report.dynamo, report.parity);
    }

    #[test]
    fn stochastic_arrival_patterns_preserve_whole_report_parity() {
        for (pattern, smoothness) in [
            (ArrivalPattern::Poisson, None),
            (ArrivalPattern::Gamma, Some(2.0)),
        ] {
            let report = run_paced_offline(
                OfflineEngineConfig {
                    topology: OfflineTopology::Aggregated,
                    workers: 2,
                    router_mode: OfflineRouterMode::Kv,
                    ..OfflineEngineConfig::default()
                },
                "model".to_string(),
                workload(9),
                pattern,
                Some(750.0),
                smoothness,
                None,
                None,
                StopConfig {
                    total_expected_requests: Some(9),
                    ..StopConfig::default()
                },
                23,
                None,
                AncillaryTimingConfig::default(),
            )
            .unwrap();

            assert_metric_parity(&report.aiperf.performance, &report.dynamo, report.parity);
        }
    }

    #[test]
    fn cancellation_is_terminal_and_byte_exact_for_every_topology() {
        let configurations = [
            OfflineEngineConfig::default(),
            OfflineEngineConfig {
                topology: OfflineTopology::Aggregated,
                workers: 2,
                router_mode: OfflineRouterMode::Kv,
                ..OfflineEngineConfig::default()
            },
            OfflineEngineConfig {
                topology: OfflineTopology::Disaggregated,
                prefill_workers: 1,
                decode_workers: 1,
                router_mode: OfflineRouterMode::Kv,
                ..OfflineEngineConfig::default()
            },
        ];
        for engine in configurations {
            let engine_debug = format!("{engine:?}");
            let report = run_paced_offline(
                engine,
                "model".to_string(),
                workload(4),
                ArrivalPattern::ConcurrencyBurst,
                None,
                None,
                Some(2),
                None,
                StopConfig {
                    total_expected_requests: Some(4),
                    ..StopConfig::default()
                },
                31,
                None,
                AncillaryTimingConfig {
                    cancellation_rate_percent: Some(100.0),
                    cancellation_delay_ns: 1,
                    ..AncillaryTimingConfig::default()
                },
            )
            .unwrap_or_else(|error| panic!("{engine_debug}: {error:#}"));

            assert_eq!(report.aiperf.performance.request_counts.num_requests, 4);
            assert_eq!(
                report.aiperf.performance.request_counts.completed_requests,
                0
            );
            assert_eq!(report.dynamo.request_counts.completed_requests, 0);
            assert_metric_parity(&report.aiperf.performance, &report.dynamo, report.parity);
        }
    }

    #[test]
    fn graph_runs_end_to_end_and_is_reproducible() {
        let run = || {
            run_graph_offline(
                OfflineEngineConfig::default(),
                BenchConfig {
                    base_urls: Vec::new(),
                    model: "model".to_string(),
                    turns: 3,
                    instances: 6,
                    workers: 1,
                    concurrency: 3,
                    max_tokens: 4,
                    request_concurrency: None,
                    prefill_concurrency: None,
                    max_duration_ns: None,
                },
            )
            .unwrap()
        };

        let first = run();
        let second = run();
        assert_eq!(first.aiperf.completed, 18);
        assert_eq!(first.aiperf.errors, 0);
        assert_eq!(first.aiperf.output_tokens, 72);
        assert!(first.aiperf.ttft_mean_ms > 0.0);
        assert_eq!(first.dynamo.request_counts.completed_requests, 18);
        assert_metric_parity(&first.performance, &first.dynamo, first.parity);
        assert_eq!(first.aiperf.completed, second.aiperf.completed);
        assert_eq!(first.aiperf.output_tokens, second.aiperf.output_tokens);
        assert_eq!(first.aiperf.ttft_mean_ms, second.aiperf.ttft_mean_ms);
        assert_eq!(
            first.performance.canonical_shared_metric_bytes().unwrap(),
            second.performance.canonical_shared_metric_bytes().unwrap()
        );
        assert_eq!(
            serde_json::to_vec(&first.dynamo).unwrap(),
            serde_json::to_vec(&second.dynamo).unwrap()
        );
    }

    #[test]
    fn every_engine_topology_completes_through_the_shared_sink() {
        let configurations = [
            OfflineEngineConfig::default(),
            OfflineEngineConfig {
                topology: OfflineTopology::Aggregated,
                workers: 2,
                ..OfflineEngineConfig::default()
            },
            OfflineEngineConfig {
                topology: OfflineTopology::Aggregated,
                workers: 2,
                router_mode: OfflineRouterMode::Kv,
                ..OfflineEngineConfig::default()
            },
            OfflineEngineConfig {
                topology: OfflineTopology::Disaggregated,
                prefill_workers: 1,
                decode_workers: 1,
                ..OfflineEngineConfig::default()
            },
            OfflineEngineConfig {
                topology: OfflineTopology::Disaggregated,
                prefill_workers: 1,
                decode_workers: 1,
                router_mode: OfflineRouterMode::Kv,
                ..OfflineEngineConfig::default()
            },
        ];

        for engine in configurations {
            let engine_debug = format!("{engine:?}");
            let report = run_paced_offline(
                engine,
                "model".to_string(),
                workload(4),
                ArrivalPattern::ConcurrencyBurst,
                None,
                None,
                Some(2),
                None,
                StopConfig {
                    total_expected_requests: Some(4),
                    ..StopConfig::default()
                },
                13,
                None,
                AncillaryTimingConfig::default(),
            )
            .unwrap_or_else(|error| panic!("{engine_debug}: {error:#}"));

            assert_eq!(
                report.aiperf.performance.request_counts.completed_requests,
                4
            );
            assert_eq!(report.dynamo.request_counts.completed_requests, 4);
            assert_metric_parity(&report.aiperf.performance, &report.dynamo, report.parity);
        }
    }

    #[test]
    fn single_turn_dataset_api_enforces_whole_report_parity() {
        use crate::dataset::{
            ComposeConfig, DatasetSource, LoadConfig, LoaderRegistry, TiktokenTokenizer,
        };
        use crate::endpoints::{
            EndpointId, EndpointRegistry, PreparedEndpointTable, RawEndpointConfig,
        };

        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let dataset = runtime
            .block_on(async {
                LoaderRegistry::with_builtin_formats()
                    .unwrap()
                    .build_dataset(
                        Some("single_turn"),
                        &LoadConfig::new(DatasetSource::Inline(serde_json::json!([{
                            "text": "hello there general",
                            "input_length": 9,
                            "output_length": 3
                        }]))),
                        &ComposeConfig::new("model", RngRoot::new(Some(1))),
                        &TiktokenTokenizer::builtin(),
                    )
                    .await
            })
            .unwrap();
        let endpoint_id = EndpointId::new("chat").unwrap();
        let endpoint = EndpointRegistry::builtin()
            .unwrap()
            .prepare(
                &endpoint_id,
                RawEndpointConfig {
                    streaming: true,
                    use_server_token_count: true,
                    ..RawEndpointConfig::default()
                },
            )
            .unwrap();
        let mut table = PreparedEndpointTable::new();
        let key = table.push(endpoint).unwrap();
        let source =
            crate::multiturn::NativeDatasetConversationSource::sequential_with_prepared_endpoint(
                dataset,
                "model",
                3,
                Rc::new(table),
                crate::multiturn::PreparedEndpointReference { key, endpoint_id },
            )
            .unwrap();
        let report = run_single_turn_dataset_offline(
            OfflineEngineConfig {
                topology: OfflineTopology::Aggregated,
                workers: 2,
                router_mode: OfflineRouterMode::Kv,
                ..OfflineEngineConfig::default()
            },
            "model".to_string(),
            Box::new(source),
            2,
            Vec::new(),
        )
        .unwrap();

        assert_eq!(
            report.aiperf.performance.request_counts.completed_requests,
            1
        );
        assert_metric_parity(&report.aiperf.performance, &report.dynamo, report.parity);
    }

    #[test]
    fn nonzero_prefix_reuse_metrics_are_byte_exact() {
        let report = run_paced_offline(
            OfflineEngineConfig {
                topology: OfflineTopology::Disaggregated,
                prefill_workers: 1,
                decode_workers: 1,
                router_mode: OfflineRouterMode::Kv,
                ..OfflineEngineConfig::default()
            },
            "model".to_string(),
            SkeletonWorkload {
                num_requests: 2,
                input_tokens: 128,
                output_tokens: 2,
                turns: 1,
                think_time_ms: None,
            },
            ArrivalPattern::Constant,
            Some(10.0),
            None,
            None,
            None,
            StopConfig {
                total_expected_requests: Some(2),
                ..StopConfig::default()
            },
            19,
            None,
            AncillaryTimingConfig::default(),
        )
        .unwrap();

        assert!(report.dynamo.prefix_cache_reused_ratio > 0.0);
        assert_metric_parity(&report.aiperf.performance, &report.dynamo, report.parity);
    }

    #[test]
    fn rejected_requests_preserve_whole_report_parity() {
        let profile = std::env::temp_dir().join(format!(
            "aiperf-offline-rejection-parity-{}.json",
            std::process::id()
        ));
        std::fs::write(&profile, br#"{"num_gpu_blocks":1}"#).unwrap();
        let result = run_paced_offline(
            OfflineEngineConfig {
                profile: Some(profile.clone()),
                ..OfflineEngineConfig::default()
            },
            "model".to_string(),
            SkeletonWorkload {
                num_requests: 3,
                input_tokens: 128,
                output_tokens: 4,
                turns: 1,
                think_time_ms: None,
            },
            ArrivalPattern::ConcurrencyBurst,
            None,
            None,
            Some(2),
            None,
            StopConfig {
                total_expected_requests: Some(3),
                ..StopConfig::default()
            },
            17,
            None,
            AncillaryTimingConfig::default(),
        );
        std::fs::remove_file(profile).unwrap();
        let report = result.unwrap();

        assert_eq!(report.aiperf.performance.request_counts.num_requests, 3);
        assert_eq!(
            report.aiperf.performance.request_counts.completed_requests,
            0
        );
        assert_metric_parity(&report.aiperf.performance, &report.dynamo, report.parity);
    }
}
