// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Reusable construction of one execution pipeline's record capture and the
//! turn dispatcher bound to it.
//!
//! The single-thread coordinator ([`super::compose_sidecars`]) and each
//! thread-per-core shard ([`super::sharding`]) previously open-coded the same
//! sequence: decide the retention mode, open the streaming per-record artifact
//! lane on the exact-fold path, build a [`RunCapture`], and wrap it in a
//! [`ConfiguredDispatcher`]. The two copies differed only in inputs, so they are
//! expressed here as one function over a request value. This module changes no
//! observable behavior; it only gives the sequence a name a streaming pipeline
//! can reuse.

use super::*;

/// The four retention decisions a run makes before it dispatches anything.
///
/// These were four adjacent positional `bool` arguments on the `RunCapture`
/// constructors; at a call site they read as `false, false, false, exact_fold`,
/// which is exactly the shape a transposition survives silently. Naming them
/// also gives a pipeline assembler one value to thread instead of four.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct RunCapturePolicy {
    /// Whether verbatim raw HTTP exchanges are retained for `raw.jsonl`.
    pub(crate) is_raw_enabled: bool,
    /// Whether a live-results sink or heartbeat lane consumes each record.
    pub(crate) needs_live_record: bool,
    /// Whether an adaptive phase consumes each completed turn's record.
    pub(crate) needs_adaptive_record: bool,
    /// Whether the run requested exact-fold retention. Sketch storage wins over
    /// this flag; see [`RunCapturePolicy::capture_mode`].
    pub(crate) is_exact_fold: bool,
}

/// The three mutually exclusive record-retention paths.
///
/// This enumerates what [`RunCapture`] already does; it introduces no fourth
/// path. `Retained` keeps every record for the batch writers, `ExactFold` folds
/// each record into an exact accumulator and drops it mid-run, and `Sketch`
/// folds into a bounded t-digest.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CaptureMode {
    /// Retain every completed record until finalization.
    Retained,
    /// Fold into an exact accumulator and drop the record mid-run.
    ExactFold,
    /// Fold into a bounded t-digest and drop the record mid-run.
    Sketch,
}

impl RunCapturePolicy {
    /// Resolve the retention path this policy selects under a storage mode.
    ///
    /// Sketch storage takes precedence, because sketch keeps a bounded t-digest
    /// and exact-fold keeps exact NaN-sparse columns and the two cannot both
    /// hold. `RunCapture` already applies that same coercion internally; this
    /// function makes it nameable so the lane and OTLP gates can be derived from
    /// the same decision the capture makes.
    pub(crate) fn capture_mode(
        self,
        storage: &crate::metrics_core::MetricsStorageMode,
    ) -> CaptureMode {
        match storage {
            crate::metrics_core::MetricsStorageMode::Sketch { .. } => CaptureMode::Sketch,
            crate::metrics_core::MetricsStorageMode::Exact if self.is_exact_fold => {
                CaptureMode::ExactFold
            }
            crate::metrics_core::MetricsStorageMode::Exact => CaptureMode::Retained,
        }
    }
}

/// Already-resolved artifact paths for the streaming per-record lane.
///
/// The caller derives these: the single-thread path resolves them against the
/// run artifact directory, and each sharded worker derives its own per-shard
/// temp path. Only the lane construction is shared.
#[derive(Clone, Debug, Default)]
pub(crate) struct RecordLanePaths {
    /// Resolved `records.jsonl` destination.
    pub(crate) records: Option<PathBuf>,
    /// Resolved `raw.jsonl` destination.
    pub(crate) raw: Option<PathBuf>,
    /// Resolved per-record CSV destination.
    pub(crate) records_csv: Option<PathBuf>,
    /// Resolved per-record Parquet destination.
    pub(crate) records_parquet: Option<PathBuf>,
    /// Resolved `outputs.json` destination.
    pub(crate) outputs: Option<PathBuf>,
    /// Whether per-record artifacts include transport-timing trace columns.
    pub(crate) include_trace: bool,
}

/// Everything one execution pipeline needs to build its capture and dispatcher.
///
/// This owns the exact inputs both existing call sites already compute. It has
/// no source, format, session, or event-time fields: the only time value it
/// carries is the monotonic run origin the callers already pass today.
pub(crate) struct CaptureServiceRequest {
    /// This pipeline's reactor-local clock.
    pub(crate) clock: Rc<dyn Clock>,
    /// Monotonic run origin shared by every phase on this timeline.
    pub(crate) origin_ns: i64,
    /// Resolved native metrics policy.
    pub(crate) metrics_config: MetricsConfig,
    /// Named retention decisions (see [`RunCapturePolicy`]).
    pub(crate) policy: RunCapturePolicy,
    /// Dispatch-ordinal authority for this pipeline.
    pub(crate) issuance: Rc<dyn IssuanceAuthority>,
    /// Per-phase global ordinal bases injected into the capture.
    pub(crate) phase_ordinal_bases: HashMap<MetricsPhase, usize>,
    /// Resolved lane destinations; a lane is opened only under exact-fold.
    pub(crate) lane_paths: RecordLanePaths,
    /// Whether native OTLP histograms fold per record at completion.
    pub(crate) is_native_otel_enabled: bool,
    /// The already-built request executor this pipeline dispatches through.
    pub(crate) execution_backend: Rc<dyn RequestExecutor>,
    /// Effective primary model stamped on each dispatched turn.
    pub(crate) model: String,
    /// This shard's executing-worker label, when the pipeline IS the worker.
    pub(crate) worker_label: Option<Arc<str>>,
}

/// The capture and the dispatcher bound to it.
pub(crate) struct PreparedCaptureService {
    /// Coordinator-side capture; the caller keeps this for finalization.
    pub(crate) capture: Rc<RunCapture>,
    /// Turn dispatcher wrapping the backend and the capture. Concrete rather
    /// than `Rc<dyn TurnDispatcher>` so the binding to `capture` stays checkable;
    /// both call sites coerce it on assignment.
    pub(crate) dispatcher: Rc<ConfiguredDispatcher>,
}

/// Builds one pipeline's [`RunCapture`] and its bound [`ConfiguredDispatcher`].
///
/// A unit struct rather than a free function so a later streaming assembler can
/// name the seam it is reusing.
pub(crate) struct CaptureService;

impl CaptureService {
    /// Build one pipeline's capture and dispatcher.
    ///
    /// Both existing call sites reduce to this. It performs no backend
    /// configuration: `set_run_origin` and `configure_measurement` stay at the
    /// call sites, inside the `async` block whose failure path still runs
    /// backend shutdown. Moving them here would skip that shutdown on their
    /// error path.
    pub(crate) fn build(request: CaptureServiceRequest) -> Result<PreparedCaptureService> {
        let is_exact_fold = matches!(
            request
                .policy
                .capture_mode(&request.metrics_config.storage_mode),
            CaptureMode::ExactFold
        );
        // The retain path uses the batch writers; only exact-fold, which drops each
        // record mid-run, needs the streaming lane. Opened here so its files are
        // truncated and its parent dirs created before dispatch, exactly as both
        // call sites do today.
        let record_lane = if is_exact_fold {
            RecordArtifactLane::new(
                request.lane_paths.records,
                request.lane_paths.raw,
                request.lane_paths.records_csv,
                request.lane_paths.records_parquet,
                request.lane_paths.outputs.clone(),
                request.lane_paths.include_trace,
            )?
        } else {
            None
        };
        // `with_worker_label(None)` and `with_otel(false)` are no-ops, so one
        // unconditional builder chain reproduces both call sites without a branch.
        let capture = Rc::new(
            RunCapture::from_policy(
                request.clock,
                request.origin_ns,
                request.metrics_config,
                request.policy,
                request.issuance,
                request.phase_ordinal_bases,
            )
            .with_record_lane(record_lane)
            .with_worker_label(request.worker_label)
            .with_otel(is_exact_fold && request.is_native_otel_enabled)
            .with_outputs_capture(is_exact_fold && request.lane_paths.outputs.is_some()),
        );
        let dispatcher = Rc::new(ConfiguredDispatcher {
            execution_backend: request.execution_backend,
            model: request.model,
            capture: capture.clone(),
        });
        Ok(PreparedCaptureService {
            capture,
            dispatcher,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clock::SimClock;
    use crate::metrics_core::MetricsStorageMode;
    use crate::multiturn::TurnToSend;

    /// Minimal executor: the capture service never dispatches through it, it only
    /// hands it to the dispatcher it builds.
    struct InertExecutor;

    #[async_trait(?Send)]
    impl RequestExecutor for InertExecutor {
        fn set_run_origin(&self, _start_ns: i64) -> Result<()> {
            Ok(())
        }

        fn inference_dimensions(&self, _turn: &TurnToSend) -> InferenceDimensions {
            InferenceDimensions::default()
        }
    }

    fn policy(is_exact_fold: bool) -> RunCapturePolicy {
        RunCapturePolicy {
            is_raw_enabled: false,
            needs_live_record: false,
            needs_adaptive_record: false,
            is_exact_fold,
        }
    }

    fn request(policy: RunCapturePolicy, lane_paths: RecordLanePaths) -> CaptureServiceRequest {
        CaptureServiceRequest {
            clock: Rc::new(SimClock::new()),
            origin_ns: 0,
            metrics_config: MetricsConfig::default(),
            policy,
            issuance: crate::engine::cellular_cell::issuance_authority_from_env(),
            phase_ordinal_bases: HashMap::new(),
            lane_paths,
            is_native_otel_enabled: false,
            execution_backend: Rc::new(InertExecutor),
            model: "test-model".to_owned(),
            worker_label: None,
        }
    }

    #[test]
    fn policy_maps_to_the_existing_capture_modes() {
        assert_eq!(
            policy(true).capture_mode(&MetricsStorageMode::Exact),
            CaptureMode::ExactFold
        );
    }

    #[test]
    fn sketch_storage_wins_over_requested_exact_fold() {
        assert_eq!(
            policy(true).capture_mode(&MetricsStorageMode::Sketch { compression: 100.0 }),
            CaptureMode::Sketch
        );
    }

    #[test]
    fn plain_exact_storage_retains_records() {
        assert_eq!(
            policy(false).capture_mode(&MetricsStorageMode::Exact),
            CaptureMode::Retained
        );
    }

    #[test]
    fn capture_service_opens_no_lane_off_the_fold_path() {
        // Every lane destination is named, yet the retain path must open none of
        // them: the lane is fold-only, as both call sites' `if exact_fold` gate
        // encodes. No file is created, so no temp directory is needed.
        let lane_paths = RecordLanePaths {
            records: Some(PathBuf::from("/nonexistent/records.jsonl")),
            raw: Some(PathBuf::from("/nonexistent/raw.jsonl")),
            records_csv: Some(PathBuf::from("/nonexistent/records.csv")),
            records_parquet: None,
            outputs: Some(PathBuf::from("/nonexistent/outputs.json")),
            include_trace: true,
        };
        let prepared = CaptureService::build(request(policy(false), lane_paths))
            .expect("retain path opens no lane and cannot fail");
        assert!(prepared.capture.record_lane.is_none());
        assert!(!prepared.capture.capture_outputs_text);
    }

    #[test]
    fn capture_service_binds_the_dispatcher_to_its_own_capture() {
        let prepared = CaptureService::build(request(policy(false), RecordLanePaths::default()))
            .expect("retain path opens no lane and cannot fail");
        assert!(Rc::ptr_eq(&prepared.capture, &prepared.dispatcher.capture));
    }
}
