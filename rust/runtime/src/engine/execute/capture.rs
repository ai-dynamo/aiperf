// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Fully-internal measurement-capture engine and record processors.

use super::*;

pub(crate) struct CaptureIdentity {
    pub(crate) uuid: Uuid,
    pub(crate) x_correlation_id: String,
    /// Coordinator-known arrival facts used to synthesize a fallback record
    /// when an identity has no drained worker record.
    pub(crate) context: MeasuredContext,
}

/// Coordinator-owned finalization facts learned after dispatch.
///
/// Worker observers record only transport facts. The coordinator uuid join applies
/// phase, session number, credit-latency policy, and terminal outcome.
pub(crate) struct CaptureLabel {
    pub(crate) phase: MetricsPhase,
    pub(crate) phase_index: Option<usize>,
    pub(crate) phase_name: Option<String>,
    pub(crate) phase_kind: Option<String>,
    pub(crate) profiling_index: Option<usize>,
    pub(crate) session_num: u64,
    pub(crate) has_credit_timestamp: bool,
    pub(crate) terminal: ReplayTerminalStatus,
    pub(crate) start_ns: i64,
    pub(crate) end_ns: i64,
}

pub(crate) struct RunCapture {
    pub(crate) clock: Rc<dyn Clock>,
    pub(crate) origin_ns: i64,
    pub(crate) metrics_config: MetricsConfig,
    pub(crate) identities: RefCell<Vec<CaptureIdentity>>,
    pub(crate) labels: RefCell<HashMap<Uuid, CaptureLabel>>,
    pub(crate) outputs: RefCell<HashMap<Uuid, CapturedModelOutput>>,
    pub(crate) raw_enabled: bool,
    pub(crate) raw_exchanges: RefCell<HashMap<Uuid, CapturedHttpExchange>>,
    /// Whether `inputs.json` is requested; gates canonical-payload retention.
    pub(crate) inputs_enabled: bool,
    /// Per-conversation canonical request bodies keyed by turn index. Retained
    /// only when `inputs_enabled`; deduplicated per `(conversation_id, turn)`
    /// (first write wins) so dataset recycling under `--request-count` collapses
    /// to one payload per dataset turn, matching `inputs.json` semantics.
    /// Sessions are emitted in conversation-id order — a stable ordering that is
    /// independent of run-to-run worker dispatch races and, for the
    /// deterministic synthetic session ids (`session_000000`, …), reproduces
    /// dataset composition order.
    pub(crate) input_sessions:
        RefCell<BTreeMap<String, BTreeMap<usize, Box<serde_json::value::RawValue>>>>,
    /// Non-consuming cloned records for the live-results sink, keyed by uuid;
    /// the authoritative record stays in the worker observer for the drain.
    pub(crate) live_records: RefCell<HashMap<Uuid, RecordIngest>>,
    /// Finished worker records staged for the adaptive window sampler, keyed by
    /// uuid. The online dispatcher records per-token facts worker-locally, so the
    /// coordinator's adaptive sampler never sees them through the callback
    /// observer; the per-phase adaptive record processor drains this map through
    /// [`AdaptiveTerminalRecordSource`]. Consumed per completed turn, so it never
    /// accumulates beyond in-flight requests.
    pub(crate) adaptive_records: RefCell<HashMap<Uuid, RecordIngest>>,
    /// Whether the live-results sink consumes each completed record.
    pub(crate) wants_live_sink_record: bool,
    /// Whether an adaptive phase consumes each completed record.
    pub(crate) wants_adaptive_record: bool,
    /// Dispatch-ordinal authority. Cellular controllers inject an autonomous issuer
    /// that stamps global ordinals spanning every cell partition.
    pub(crate) issuance: Rc<dyn IssuanceAuthority>,
    /// Global cumulative dispatch count of the phases before each phase (0 for the
    /// first). A cell's per-phase sampler restarts at 0, so the autonomous issuer
    /// adds this base to a turn's phase-local slot to recover the single-cell
    /// absolute slot. Empty (all-zero) for the single-process path.
    pub(crate) phase_ordinal_bases: HashMap<MetricsPhase, usize>,
    /// Whether this capture runs in metrics-only (sketch) mode: each completed
    /// turn's record is folded into `accumulator` and dropped as the run streams,
    /// so peak coordinator memory is O(sketch) instead of O(records).
    pub(crate) metrics_only: bool,
    /// Bounded streaming accumulator that folds each metrics-only record on
    /// completion (see [`RunCapture::fold_streaming`]). Empty and unused in exact
    /// mode. `RefCell` because the fold runs from `&self` record processing.
    pub(crate) accumulator: RefCell<MetricsAccumulator>,
    /// Errored/canceled metrics-only records retained for the report's error
    /// grouping ([`group_record_errors`]); the fold drops every non-errored
    /// record, so this stays O(errors), not O(records).
    pub(crate) streaming_errored: RefCell<Vec<CapturedRecord>>,
    /// Whether this capture runs in exact-fold mode: like sketch, it folds each
    /// completed record into `accumulator` and drops the heavy per-record data as
    /// the run streams, BUT the accumulator stays in EXACT (non-sketch) storage and
    /// each record is stamped with its absolute dispatch `request_index` before the
    /// fold, so `export_results` yields exact percentiles/timeslices/series — not the
    /// sketch approximation. Distinct from `metrics_only` (sketch); the two are
    /// mutually exclusive. Selected only for the single-thread `DirectIssuanceAuthority`
    /// scheduled path with no per-record file artifacts (see [`exact_fold_eligible`]).
    pub(crate) exact_fold: bool,
    /// Monotonic dispatch-ordinal counter for exact-fold, incremented once per
    /// [`RunCapture::begin`]. Its value at `begin` is the turn's `flat_local` — the
    /// dense absolute record slot the [`DirectIssuanceAuthority`] would stamp — so
    /// exact-fold rows land at the same absolute ordinals as the retained-record path.
    /// Unused (and left at 0) in sketch/exact modes.
    pub(crate) fold_dispatch_next: Cell<usize>,
    /// Maps each dispatched turn's uuid to the dispatch ordinal assigned at `begin`,
    /// consumed once at completion by the phase processor's exact-fold branch. Drained
    /// per completed turn, so it never outgrows in-flight work. Only populated in
    /// exact-fold mode.
    pub(crate) fold_dispatch_ordinals: RefCell<HashMap<Uuid, usize>>,
    /// Streaming per-record artifact lane: when exact-fold runs a records/
    /// raw/CSV-artifact run, each completed record's rows are appended here before the
    /// fold drops it, so the artifacts are still emitted without retaining every record.
    /// `None` on the retained-record path (which uses the batch writers) and whenever no
    /// lane artifact is requested. Set once at construction via [`Self::with_record_lane`].
    pub(crate) record_lane: Option<Rc<RecordArtifactLane>>,
    /// Per-record OTLP histogram accumulator: when native OTLP is enabled on
    /// the exact-fold path, each completed profiling record is folded here at
    /// completion (an order-independent fold) and then dropped, instead of iterating
    /// the retained record set post-run. `None` when native OTLP is off. Set once at
    /// construction via [`Self::with_otel`].
    pub(crate) otel: Option<RefCell<OtelRecordAccumulator>>,
    /// Whether the fold-and-drop path must retain each turn's model output text long
    /// enough to stream its `outputs.json` entry: `record_model_output`
    /// stages the text in `outputs`, [`Self::fold_record`] attaches it to the streamed
    /// record and drops it. `false` on the retain path (which keeps `outputs` for the
    /// batch writer) and whenever no `outputs.json` artifact is requested. Set once via
    /// [`Self::with_outputs_capture`].
    pub(crate) capture_outputs_text: bool,
}

impl RunCapture {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        clock: Rc<dyn Clock>,
        origin_ns: i64,
        config: MetricsConfig,
        raw_enabled: bool,
        inputs_enabled: bool,
        wants_live_sink_record: bool,
        wants_adaptive_record: bool,
        exact_fold: bool,
    ) -> Self {
        // Cell processes select the autonomous issuer from `AIPERF_CELL_ID` and
        // `AIPERF_CELL_COUNT`; single-process execution uses direct issuance.
        Self::new_with_issuance(
            clock,
            origin_ns,
            config,
            raw_enabled,
            inputs_enabled,
            wants_live_sink_record,
            wants_adaptive_record,
            exact_fold,
            crate::engine::cellular_cell::issuance_authority_from_env(),
        )
    }

    /// Construct with an explicitly injected dispatch-ordinal issuer. Thread-per-core
    /// execution builds one `RunCapture` per sub-cell thread with a per-thread issuer
    /// (see
    /// [`issuance_authority_for`](crate::engine::cellular_cell::issuance_authority_for))
    /// whose `(cell_id, cell_count)` partition the process-global env vars cannot
    /// express. Per-phase ordinal bases come from the environment and carry no
    /// partition.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_with_issuance(
        clock: Rc<dyn Clock>,
        origin_ns: i64,
        config: MetricsConfig,
        raw_enabled: bool,
        inputs_enabled: bool,
        wants_live_sink_record: bool,
        wants_adaptive_record: bool,
        exact_fold: bool,
        issuance: Rc<dyn IssuanceAuthority>,
    ) -> Self {
        // Cell processes read global phase ordinal bases from
        // `AIPERF_CELL_PHASE_ORDINAL_BASES`; single-process execution uses zero bases.
        Self::new_with_issuance_and_bases(
            clock,
            origin_ns,
            config,
            raw_enabled,
            inputs_enabled,
            wants_live_sink_record,
            wants_adaptive_record,
            exact_fold,
            issuance,
            crate::engine::cellular_cell::phase_ordinal_bases_from_env(),
        )
    }

    /// Construct with explicitly injected per-phase global ordinal bases.
    ///
    /// A single-process thread-per-core scheduled run (cells == 1, no controller)
    /// has no `AIPERF_CELL_PHASE_ORDINAL_BASES` env var, yet its `W` sub-cell
    /// threads still need each phase's global base so profiling ordinals do not
    /// collide with warmup's `[0, W)` block. The sharded runtime computes the
    /// bases from the phase `requests` budgets (the same policy as
    /// [`crate::engine::cellular_controller::phase_ordinal_bases`]) and injects the same
    /// map into every thread's capture. Controller children use the global,
    /// partition-independent environment bases for every thread.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_with_issuance_and_bases(
        clock: Rc<dyn Clock>,
        origin_ns: i64,
        config: MetricsConfig,
        raw_enabled: bool,
        inputs_enabled: bool,
        wants_live_sink_record: bool,
        wants_adaptive_record: bool,
        exact_fold: bool,
        issuance: Rc<dyn IssuanceAuthority>,
        phase_ordinal_bases: HashMap<MetricsPhase, usize>,
    ) -> Self {
        // Sketch storage mode selects metrics-only fold-and-drop; exact mode does not
        // use the streaming fields.
        let metrics_only = matches!(
            config.storage_mode,
            crate::metrics_core::MetricsStorageMode::Sketch { .. }
        );
        // Sketch and exact-fold are mutually exclusive: sketch keeps the bounded
        // t-digest, exact-fold keeps exact NaN-sparse columns. The caller only sets
        // exact_fold on the exact (non-sketch) scheduled path, but guard it so a
        // sketch config can never accidentally run the exact-fold column path.
        let exact_fold = exact_fold && !metrics_only;
        let accumulator = RefCell::new(MetricsAccumulator::with_config(config.clone()));
        Self {
            clock,
            origin_ns,
            metrics_config: config,
            identities: RefCell::new(Vec::new()),
            labels: RefCell::new(HashMap::new()),
            outputs: RefCell::new(HashMap::new()),
            raw_enabled,
            raw_exchanges: RefCell::new(HashMap::new()),
            inputs_enabled,
            input_sessions: RefCell::new(BTreeMap::new()),
            live_records: RefCell::new(HashMap::new()),
            adaptive_records: RefCell::new(HashMap::new()),
            wants_live_sink_record,
            wants_adaptive_record,
            issuance,
            phase_ordinal_bases,
            metrics_only,
            accumulator,
            streaming_errored: RefCell::new(Vec::new()),
            exact_fold,
            fold_dispatch_next: Cell::new(0),
            fold_dispatch_ordinals: RefCell::new(HashMap::new()),
            record_lane: None,
            otel: None,
            capture_outputs_text: false,
        }
    }

    /// Attach the streaming per-record artifact lane, consumed once per completed
    /// record in the exact-fold [`Self::fold_record`] path before the record is
    /// dropped. Builder-style so only the single-thread exact-fold call site opts in;
    /// every other construction leaves it `None` and uses the batch writers.
    pub(crate) fn with_record_lane(mut self, lane: Option<Rc<RecordArtifactLane>>) -> Self {
        self.record_lane = lane;
        self
    }

    /// Enable per-record OTLP folding at completion. When `enabled`, each
    /// completed profiling record is folded into a bounded [`OtelRecordAccumulator`]
    /// in [`Self::fold_record`] and dropped, so the OTLP histograms need no retained
    /// record set. Builder-style so only the exact-fold call site with native OTLP
    /// opts in; every other construction leaves it `None` and the retain path folds
    /// the retained records post-run.
    pub(crate) fn with_otel(mut self, enabled: bool) -> Self {
        if enabled {
            self.otel = Some(RefCell::new(OtelRecordAccumulator::new()));
        }
        self
    }

    /// Retain each turn's model output text for streaming `outputs.json`.
    /// When `enabled`, `record_model_output` stages the text even on the fold-and-drop
    /// path so [`Self::fold_record`] can attach it to the streamed record before the
    /// fold drops it. Builder-style so only the exact-fold call site with an
    /// `outputs.json` artifact opts in.
    pub(crate) fn with_outputs_capture(mut self, enabled: bool) -> Self {
        self.capture_outputs_text = enabled;
        self
    }

    /// Move the folded per-record OTLP accumulator out for the finalize, if one was
    /// attached. Consumed once at run end (leaves an empty accumulator behind).
    pub(crate) fn take_otel(&self) -> Option<OtelRecordAccumulator> {
        self.otel
            .as_ref()
            .map(|cell| std::mem::take(&mut *cell.borrow_mut()))
    }

    /// Flush and close the streaming per-record artifact lane, if one is attached.
    /// Called once at run end after every record has been folded (and its rows
    /// streamed); a lazy CSV that saw no non-skipped row stays absent.
    pub(crate) fn finish_record_lane(&self) -> Result<()> {
        match &self.record_lane {
            Some(lane) => lane.finish(),
            None => Ok(()),
        }
    }

    /// Retain one turn's canonical request body for `inputs.json`.
    ///
    /// Called on the coordinator thread for every dispatched turn (independent
    /// of raw-artifact capture). Deduplicates per `(conversation_id, turn_index)`
    /// so recycled dataset turns collapse to a single payload, and remembers
    /// first-dispatch conversation order for deterministic session ordering.
    pub(crate) fn record_input_payload(
        &self,
        conversation_id: &str,
        turn_index: usize,
        payload: &[u8],
    ) -> Result<()> {
        if !self.inputs_enabled {
            return Ok(());
        }
        let mut sessions = self.input_sessions.borrow_mut();
        let turns = sessions.entry(conversation_id.to_string()).or_default();
        if let std::collections::btree_map::Entry::Vacant(slot) = turns.entry(turn_index) {
            let parsed: Box<serde_json::value::RawValue> = serde_json::from_slice(payload)
                .with_context(|| {
                    format!(
                        "validating canonical request payload for inputs.json \
                         (conversation {conversation_id}, turn {turn_index})"
                    )
                })?;
            slot.insert(parsed);
        }
        Ok(())
    }

    /// Consume the retained payloads into conversation-id-ordered
    /// `inputs.json` sessions. The `BTreeMap` iteration yields sorted keys, so
    /// ordering is identical across same-seed runs regardless of worker races.
    pub(crate) fn take_input_sessions(&self) -> Vec<InputSession> {
        self.input_sessions
            .take()
            .into_iter()
            .map(|(session_id, turns)| InputSession {
                session_id,
                payloads: turns.into_values().collect(),
            })
            .collect()
    }

    /// Whether the worker should return a per-turn record: a non-consuming
    /// snapshot for the live-results sink or the adaptive window sampler, or a
    /// consuming drain in metrics-only mode (see [`MeasuredContext::consume_record`]).
    pub(crate) fn wants_live_record(&self) -> bool {
        self.wants_live_sink_record || self.wants_adaptive_record || self.folds_records()
    }

    /// Whether this capture folds each completed record into `accumulator` and drops
    /// the heavy per-record data (worker `token_arrivals_ns`, identities/labels) as
    /// the run streams, rather than retaining every record until end-of-run. True for
    /// both sketch (`metrics_only`) and exact-fold; the two differ only in the
    /// accumulator's storage mode and whether `request_index` is stamped.
    pub(crate) fn folds_records(&self) -> bool {
        self.metrics_only || self.exact_fold
    }

    /// Assign and record the next dense dispatch ordinal for `uuid` (exact-fold only).
    /// Called once per turn at [`Self::begin`]; the value is the turn's `flat_local`,
    /// which the [`DirectIssuanceAuthority`] maps identically to its `request_index`.
    pub(crate) fn assign_fold_ordinal(&self, uuid: Uuid) -> usize {
        let ordinal = self.fold_dispatch_next.get();
        self.fold_dispatch_next.set(ordinal + 1);
        self.fold_dispatch_ordinals
            .borrow_mut()
            .insert(uuid, ordinal);
        ordinal
    }

    /// Consume the dispatch ordinal assigned to `uuid` at `begin`. `None` for a turn
    /// no `begin` recorded (never happens on the exact-fold path, where every
    /// dispatched turn passes through `begin`), in which case the fold appends.
    pub(crate) fn take_fold_ordinal(&self, uuid: Uuid) -> Option<usize> {
        self.fold_dispatch_ordinals.borrow_mut().remove(&uuid)
    }

    /// Record the dispatch identity plus coordinator-known arrival facts, and
    /// return the measured context the dispatcher forwards to the worker so it
    /// registers arrival locally. The identity push order is the global dispatch
    /// ordinal used at finish; it runs on the coordinator thread before backend
    /// dispatch, so it is independent of worker count.
    pub(crate) fn begin(&self, turn: &TurnToSend) -> MeasuredContext {
        let arrival_ms = self.clock.now_ns().saturating_sub(self.origin_ns) as f64 / 1_000_000.0;
        let context = MeasuredContext {
            arrival_ms,
            input_length: turn.input_length,
            requested_output_length: turn.max_output_tokens,
            metadata: RequestMetricMetadata {
                turn_index: u32::try_from(turn.turn_index).unwrap_or(u32::MAX),
                conversation_id: Some(turn.conversation_id.clone()),
                audio_duration_s: turn.audio_duration_seconds,
                ..RequestMetricMetadata::default()
            },
            wants_live_record: self.wants_live_record(),
            // Fold-and-drop modes (sketch + exact-fold) fold each record and drop it,
            // so the worker must move the record out of its observer to free token
            // storage as it goes.
            consume_record: self.folds_records(),
        };
        // Fold-and-drop modes never join records by dispatch identity at finish (they
        // fold each on completion), so skip the O(records) identity retention — the
        // fold source is the per-turn record staged by `record_live`, and a failed
        // turn's record is synthesized in `process` instead. Other modes retain the
        // identity for the finish-time UUID join.
        if !self.folds_records() {
            self.identities.borrow_mut().push(CaptureIdentity {
                uuid: turn.uuid,
                x_correlation_id: turn.x_correlation_id.clone(),
                context: context.clone(),
            });
        }
        // Exact-fold stamps each record's absolute dispatch `request_index` so its row
        // lands at the same ordinal the retained-record path would assign. The ordinal is
        // this turn's `begin` push order (dense `0, 1, 2, …`), matching the
        // `DirectIssuanceAuthority` `flat_local`; record it here for the completion-time
        // fold. (Sketch ignores `request_index`, so it needs no ordinal.)
        if self.exact_fold {
            self.assign_fold_ordinal(turn.uuid);
        }
        context
    }

    pub(crate) fn label(
        &self,
        credit: &IssuedCredit,
        phase: MetricsPhase,
        identity: &PhaseIdentity,
        has_credit_timestamp: bool,
        outcome: &TurnDispatchOutcome,
    ) {
        // Labels feed the finish-time uuid join only; fold-and-drop modes fold each
        // record on completion and never join, so retaining them would be pure
        // O(records) waste. The fold applies phase/session/admit itself.
        if self.folds_records() {
            return;
        }
        self.labels.borrow_mut().insert(
            credit.turn.uuid,
            CaptureLabel {
                phase,
                phase_index: Some(identity.phase_index),
                phase_name: Some(identity.phase_name.clone()),
                phase_kind: Some(match identity.phase_kind {
                    PhaseKind::Warmup => "warmup".to_string(),
                    PhaseKind::Profiling => "profiling".to_string(),
                }),
                profiling_index: identity.profiling_index,
                session_num: credit.id,
                has_credit_timestamp,
                terminal: outcome.terminal,
                start_ns: outcome.start_ns,
                end_ns: outcome.end_ns,
            },
        );
    }

    pub(crate) fn record_live(&self, uuid: Uuid, record: RecordIngest) {
        // Fold-and-drop modes stage every completed turn's record for the phase
        // processor's fold-and-drop, regardless of any live/adaptive consumer. An
        // adaptive phase still needs its own copy (read-only window sampling), so
        // clone into `adaptive_records` when one is active. Both maps are drained
        // per completed turn, so neither outgrows in-flight work.
        if self.folds_records() {
            if self.wants_adaptive_record {
                self.adaptive_records
                    .borrow_mut()
                    .insert(uuid, record.clone());
            }
            self.live_records.borrow_mut().insert(uuid, record);
            return;
        }
        // Fan the worker's non-consuming snapshot out to each interested
        // consumer. Both drain their own map per completed turn, so neither
        // consumer starves the other and neither map outgrows in-flight work.
        match (self.wants_adaptive_record, self.wants_live_sink_record) {
            (true, true) => {
                self.adaptive_records
                    .borrow_mut()
                    .insert(uuid, record.clone());
                self.live_records.borrow_mut().insert(uuid, record);
            }
            (true, false) => {
                self.adaptive_records.borrow_mut().insert(uuid, record);
            }
            (false, true) => {
                self.live_records.borrow_mut().insert(uuid, record);
            }
            (false, false) => {}
        }
    }

    pub(crate) fn record_model_output(
        &self,
        uuid: Uuid,
        flattened_text: &str,
        visible_text: Option<&str>,
        reasoning_text: Option<&str>,
    ) -> Result<()> {
        // Fold-and-drop modes retain no per-record output artifact by default (sketch
        // forbids `outputs_path` in `validate_plan`), so keeping the text would be pure
        // O(records) waste; drop it before the map grows. The exception is exact-fold
        // with a streaming `outputs.json`: stage the text so `fold_record` can
        // attach it to the streamed entry and drop it per completion (bounded to
        // in-flight work).
        if self.folds_records() && !self.capture_outputs_text {
            return Ok(());
        }
        ensure!(
            self.outputs
                .borrow_mut()
                .insert(
                    uuid,
                    CapturedModelOutput::from_parts(flattened_text, visible_text, reasoning_text),
                )
                .is_none(),
            "native model output was recorded more than once for request {uuid}"
        );
        Ok(())
    }

    pub(crate) fn record_http_exchange(
        &self,
        uuid: Uuid,
        request_payload: bytes::Bytes,
        record: crate::transport::core::RequestRecord,
    ) -> Result<()> {
        // Copy the (potentially multi-MB) body into an owned buffer ONLY when a
        // raw artifact will consume it. On the default path this returns before
        // touching the bytes, so a 24 MB image body stays a shared refcount and
        // is never duplicated per request.
        if !self.raw_enabled {
            return Ok(());
        }
        ensure!(
            self.raw_exchanges
                .borrow_mut()
                .insert(
                    uuid,
                    CapturedHttpExchange {
                        request_payload: request_payload.to_vec(),
                        record,
                    },
                )
                .is_none(),
            "native HTTP exchange was recorded more than once for request {uuid}"
        );
        Ok(())
    }

    /// Build a live-sink record from the worker's non-consuming cloned record.
    ///
    /// The clone is removed from the pending map (each request emits once); the
    /// authoritative record stays in the worker observer for the final drain, so
    /// live emission never undercounts the end-of-run aggregate. `admit_ns` and
    /// `session_num` are patched to the credit-issued values the live consumer
    /// expects.
    pub(crate) fn snapshot_live(&self, credit: &IssuedCredit) -> Option<CapturedRecord> {
        let mut ingest = self.live_records.borrow_mut().remove(&credit.turn.uuid)?;
        ingest.session_num = credit.id;
        if ingest.admit_ns.is_some() {
            ingest.admit_ns = Some(credit.issued_ns.saturating_sub(self.origin_ns));
        }
        Some(CapturedRecord {
            uuid: credit.turn.uuid,
            x_correlation_id: credit.turn.x_correlation_id.clone(),
            output: self
                .outputs
                .borrow()
                .get(&credit.turn.uuid)
                .cloned()
                .unwrap_or_default(),
            raw: None,
            ingest,
        })
    }

    /// Join per-worker drained records to dispatch identities and produce the
    /// coordinator's captured records in dispatch order.
    ///
    /// Worker observers accumulate per-worker in dense-local order, so a record's
    /// drain slot is meaningless globally. Keyed on each record's true drain
    /// `Uuid` (never `correlation_id`, which aggregate-only mode blanks), this:
    ///
    /// 1. resolves each identity to its worker record, or synthesizes a fallback
    ///    record for an identity that failed before any worker touched it;
    /// 2. stamps `request_index` to the identity's global dispatch ordinal so the
    ///    downstream re-ingest lands each record at a unique, dense dispatch-ordered
    ///    slot;
    /// 3. patches `phase`/`session_num`/`admit_ns` from the coordinator-owned
    ///    label, preserving the credit-latency time base.
    pub(crate) fn finish(
        &self,
        issued_times: &HashMap<Uuid, i64>,
        drained: Vec<(Uuid, RecordIngest)>,
    ) -> Result<Vec<CapturedRecord>> {
        let identities = self.identities.borrow();
        let labels = self.labels.borrow();
        let outputs = self.outputs.borrow();
        let mut raw_exchanges = self.raw_exchanges.take();

        let mut records_by_uuid = self.resolve_records_by_uuid(&identities, &labels, drained)?;

        // Emit rows in dispatch (identity) order. `ordinal` (begin order) is the
        // cumulative flat dispatch index; `phase_counters` tracks the per-phase
        // dispatch index because a cell's sampler restarts each phase, so the
        // cellular issuer's ordinal must be phase-local. The identity issuer uses the
        // flat ordinal.
        let mut phase_counters: HashMap<_, usize> = HashMap::new();
        identities
            .iter()
            .enumerate()
            .map(|(ordinal, identity)| {
                let mut ingest = records_by_uuid.remove(&identity.uuid).ok_or_else(|| {
                    anyhow!(
                        "captured request {} produced no native metric record",
                        identity.uuid
                    )
                })?;
                self.patch_joined_ingest(
                    ordinal,
                    identity,
                    &labels,
                    issued_times,
                    &mut phase_counters,
                    &mut ingest,
                )?;
                Ok(CapturedRecord {
                    uuid: identity.uuid,
                    x_correlation_id: identity.x_correlation_id.clone(),
                    output: outputs.get(&identity.uuid).cloned().unwrap_or_default(),
                    raw: raw_exchanges.remove(&identity.uuid),
                    ingest,
                })
            })
            .collect()
    }

    /// Fold one completed fold-and-drop turn's record into the streaming
    /// accumulator and drop it, keeping peak memory O(sketch)/O(scalars) rather than
    /// O(full records).
    ///
    /// Applies the coordinator-owned fields the finish-time join would apply —
    /// `phase` (the worker defaults every record, warmup included, to Profiling),
    /// `session_num`, and the credit-issued `admit_ns` (bit-equal to the finish
    /// path's `issued_offset_ns` because the run origin equals every phase's start).
    /// In exact-fold `request_index` is `Some` (the turn's dense absolute dispatch
    /// ordinal), so the record's row lands at the same absolute slot as retained-record
    /// execution and exact percentiles/timeslices/series are byte-identical; in
    /// sketch it is `None` (the sketch store ignores it).
    ///
    /// Sketch approximate-memory contract: the sketch (t-digest percentiles, Welford
    /// mean/M2, and the float running sums) is order-independent only up to a few
    /// ULPs, and this folds in *completion* order rather than the finish path's
    /// *dispatch* order. So percentiles, means, and float sums drift a few ULPs —
    /// below display precision — and lose exact run-to-run reproducibility, while
    /// counts, min/max, and integer sums stay bit-identical. Exact-fold does NOT drift:
    /// `insert_record_at(request_index)` places each row at its absolute slot, so the
    /// dense NaN-sparse columns are byte-identical to dispatch-order ingestion. That
    /// sketch drift is the accepted price of bounded memory; a reorder buffer would be
    /// O(records) and defeat the goal.
    pub(crate) fn fold_streaming(
        &self,
        ingest: RecordIngest,
        phase: MetricsPhase,
        identity: &PhaseIdentity,
        has_credit_timestamp: bool,
        request_index: Option<usize>,
        credit: &IssuedCredit,
    ) -> Result<()> {
        let admit_ns =
            has_credit_timestamp.then(|| credit.issued_ns.saturating_sub(self.origin_ns));
        self.fold_record(
            ingest,
            credit.turn.uuid,
            &credit.turn.x_correlation_id,
            phase,
            Some(identity.phase_index),
            Some(identity.phase_name.clone()),
            Some(match identity.phase_kind {
                PhaseKind::Warmup => "warmup".to_string(),
                PhaseKind::Profiling => "profiling".to_string(),
            }),
            identity.profiling_index,
            credit.id,
            admit_ns,
            request_index,
        )
    }

    /// Stamp the coordinator-owned fields onto one completed record, process it into
    /// the streaming accumulator, and drop it — retaining only errored/canceled
    /// records for [`group_record_errors`]. The primitive both the sketch and
    /// exact-fold fold paths share; taking the fields directly (not an
    /// [`IssuedCredit`]) keeps it unit-testable in isolation.
    ///
    /// `request_index` overwrites the worker's dense-local slot with the absolute
    /// dispatch ordinal in exact-fold; passing `None` (sketch) leaves the worker value
    /// untouched, which the sketch store ignores.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn fold_record(
        &self,
        mut ingest: RecordIngest,
        uuid: Uuid,
        x_correlation_id: &str,
        phase: MetricsPhase,
        phase_index: Option<usize>,
        phase_name: Option<String>,
        phase_kind: Option<String>,
        profiling_index: Option<usize>,
        session_num: u64,
        admit_ns: Option<i64>,
        request_index: Option<usize>,
    ) -> Result<()> {
        ingest.phase = phase;
        ingest.phase_index = phase_index;
        ingest.phase_name = phase_name;
        ingest.phase_kind = phase_kind;
        ingest.profiling_index = profiling_index;
        ingest.session_num = session_num;
        ingest.admit_ns = admit_ns;
        if let Some(row) = request_index {
            ingest.request_index = Some(row);
        }
        self.accumulator.borrow_mut().process_record(&ingest);
        let errored = ingest.errored || ingest.canceled;
        // Per-record OTLP folds every PROFILING record (success and error
        // alike, matching the retain path's post-run loop) into the order-independent
        // accumulator; warmup records never contribute.
        let wants_otel = self.otel.is_some() && phase == MetricsPhase::Profiling;
        // Materialize a CapturedRecord only when something consumes it: the streaming
        // artifact lane writes every record's rows, the per-record OTLP fold
        // observes each profiling record, and the error grouping retains
        // errored records. The raw HTTP exchange captured for this uuid is pulled out
        // here (present only when `raw_path` is enabled) so raw.jsonl and the error
        // classification see the same transport facts the retain path does; the drop
        // keeps `raw_exchanges` bounded to in-flight work.
        if self.record_lane.is_some() || errored || wants_otel {
            let raw = self.raw_exchanges.borrow_mut().remove(&uuid);
            // The streaming outputs.json entry reads the model output text;
            // drain the text staged by `record_model_output` so the entry carries it,
            // then the record (and its text) is dropped here. records/raw/CSV rows never
            // read it, so the default is byte-safe when outputs.json is not requested.
            let output = if self.capture_outputs_text {
                self.outputs.borrow_mut().remove(&uuid).unwrap_or_default()
            } else {
                CapturedModelOutput::default()
            };
            let captured = CapturedRecord {
                uuid,
                x_correlation_id: x_correlation_id.to_string(),
                output,
                raw,
                ingest,
            };
            if let Some(lane) = &self.record_lane {
                lane.write(&captured, &self.metrics_config)?;
            }
            if wants_otel && let Some(cell) = &self.otel {
                observe_otel_record(&mut cell.borrow_mut(), &captured, &self.metrics_config);
            }
            if errored {
                self.streaming_errored.borrow_mut().push(captured);
            }
        }
        Ok(())
    }

    /// Remove one metrics-only turn's staged worker record for the phase
    /// processor's fold. Absent for a turn that failed or was canceled before it
    /// completed (its `Err`/cancel path never called `record_live`), in which case
    /// the processor synthesizes the record instead.
    pub(crate) fn take_streaming_record(&self, uuid: Uuid) -> Option<RecordIngest> {
        self.live_records.borrow_mut().remove(&uuid)
    }

    /// Synthesize the record for a metrics-only turn the worker never staged — a
    /// dispatch that failed or was canceled before completion.
    ///
    /// Built exactly as [`resolve_records_by_uuid`]'s finish-time fallback: a
    /// one-shot [`NativeMetricsObserver`] fed the same arrival/terminal/response
    /// facts (from `credit.turn` and `outcome`), so the errored/canceled flag,
    /// `ErrorRequestCount`, and error grouping match the exact and retained paths.
    /// `fold_streaming` then applies phase/session/admit.
    pub(crate) fn synthesize_streaming_fallback(
        &self,
        credit: &IssuedCredit,
        outcome: &TurnDispatchOutcome,
    ) -> RecordIngest {
        let turn = &credit.turn;
        let fallback = NativeMetricsObserver::new(
            self.clock.clone(),
            self.origin_ns,
            self.metrics_config.clone(),
        );
        let arrival_ms = self.clock.now_ns().saturating_sub(self.origin_ns) as f64 / 1_000_000.0;
        fallback.register_metadata(
            turn.uuid,
            RequestMetricMetadata {
                turn_index: u32::try_from(turn.turn_index).unwrap_or(u32::MAX),
                conversation_id: Some(turn.conversation_id.clone()),
                audio_duration_s: turn.audio_duration_seconds,
                ..RequestMetricMetadata::default()
            },
        );
        fallback.on_arrival(
            turn.uuid,
            arrival_ms,
            turn.input_length,
            turn.max_output_tokens,
        );
        fallback.on_terminal(turn.uuid, outcome.terminal);
        fallback.record_response(
            turn.uuid,
            NativeResponseMetadata {
                start_ns: Some(outcome.start_ns),
                end_ns: Some(outcome.end_ns),
                ..NativeResponseMetadata::default()
            },
        );
        fallback
            .finish_with_records()
            .records
            .into_iter()
            .find_map(|(uuid, ingest)| (uuid == turn.uuid).then_some(ingest))
            .unwrap_or_else(|| {
                // Defensive: the observer always yields the record it was just fed;
                // fall back to a minimal terminal record rather than panicking.
                let mut ingest = RecordIngest::minimal(
                    outcome.start_ns,
                    outcome.end_ns,
                    MetricsPhase::Profiling,
                );
                ingest.errored = matches!(
                    outcome.terminal,
                    ReplayTerminalStatus::Failed | ReplayTerminalStatus::Rejected
                );
                ingest.canceled = outcome.terminal == ReplayTerminalStatus::Canceled;
                ingest
            })
    }

    /// Move the streaming accumulator and its retained errored records out for the
    /// finalize, leaving a fresh empty accumulator behind so the capture stays
    /// reusable. The sharded path ships the returned accumulator as its shard
    /// partition; the single-thread path merges it into the report accumulator.
    pub(crate) fn take_streamed(&self) -> (MetricsAccumulator, Vec<CapturedRecord>) {
        let accumulator = std::mem::replace(
            &mut *self.accumulator.borrow_mut(),
            MetricsAccumulator::with_config(self.metrics_config.clone()),
        );
        (accumulator, self.streaming_errored.take())
    }

    /// Builds the uuid→record map for the exact-mode finish, synthesizing fallback
    /// records for identities no worker observer produced. (Metrics-only mode folds
    /// each record on completion and synthesizes its own per-turn fallback in
    /// [`RunCapture::synthesize_streaming_fallback`], so it never joins here.)
    pub(crate) fn resolve_records_by_uuid(
        &self,
        identities: &[CaptureIdentity],
        labels: &HashMap<Uuid, CaptureLabel>,
        drained: Vec<(Uuid, RecordIngest)>,
    ) -> Result<HashMap<Uuid, RecordIngest>> {
        let mut records_by_uuid: HashMap<Uuid, RecordIngest> =
            HashMap::with_capacity(drained.len());
        for (uuid, ingest) in drained {
            ensure!(
                records_by_uuid.insert(uuid, ingest).is_none(),
                "worker drained request {uuid} more than once"
            );
        }
        let missing: Vec<&CaptureIdentity> = identities
            .iter()
            .filter(|identity| !records_by_uuid.contains_key(&identity.uuid))
            .collect();
        if !missing.is_empty() {
            let fallback = NativeMetricsObserver::new(
                self.clock.clone(),
                self.origin_ns,
                self.metrics_config.clone(),
            );
            for identity in &missing {
                let label = labels.get(&identity.uuid);
                let terminal = label
                    .map(|label| label.terminal)
                    .unwrap_or(ReplayTerminalStatus::Failed);
                fallback.register_metadata(identity.uuid, identity.context.metadata.clone());
                fallback.on_arrival(
                    identity.uuid,
                    identity.context.arrival_ms,
                    identity.context.input_length,
                    identity.context.requested_output_length,
                );
                fallback.on_terminal(identity.uuid, terminal);
                let (start_ns, end_ns) = label
                    .map(|label| (label.start_ns, label.end_ns))
                    .unwrap_or_else(|| {
                        let now = self.clock.now_ns();
                        (now, now)
                    });
                fallback.record_response(
                    identity.uuid,
                    NativeResponseMetadata {
                        start_ns: Some(start_ns),
                        end_ns: Some(end_ns),
                        ..NativeResponseMetadata::default()
                    },
                );
            }
            for (uuid, ingest) in fallback.finish_with_records().records {
                records_by_uuid.insert(uuid, ingest);
            }
        }
        ensure!(
            records_by_uuid.len() == identities.len(),
            "native record capture finalized {} records for {} dispatched identities",
            records_by_uuid.len(),
            identities.len()
        );
        Ok(records_by_uuid)
    }

    /// Patches one joined record's coordinator-owned fields (phase, session number,
    /// global dispatch ordinal, and admit timestamp) exactly as [`finish`] does.
    pub(crate) fn patch_joined_ingest(
        &self,
        ordinal: usize,
        identity: &CaptureIdentity,
        labels: &HashMap<Uuid, CaptureLabel>,
        issued_times: &HashMap<Uuid, i64>,
        phase_counters: &mut HashMap<MetricsPhase, usize>,
        ingest: &mut RecordIngest,
    ) -> Result<()> {
        let has_credit_timestamp = labels
            .get(&identity.uuid)
            .map(|label| label.has_credit_timestamp)
            .unwrap_or(true);
        if let Some(label) = labels.get(&identity.uuid) {
            ingest.phase = label.phase;
            ingest.phase_index = label.phase_index;
            ingest.phase_name = label.phase_name.clone();
            ingest.phase_kind = label.phase_kind.clone();
            ingest.profiling_index = label.profiling_index;
            ingest.session_num = label.session_num;
        }
        let within_phase = phase_counters.entry(ingest.phase).or_insert(0);
        let within = *within_phase;
        *within_phase += 1;
        let phase_base = self
            .phase_ordinal_bases
            .get(&ingest.phase)
            .copied()
            .unwrap_or(0);
        ingest.request_index = Some(self.issuance.global_ordinal(ordinal, phase_base, within));
        ingest.admit_ns = if has_credit_timestamp {
            Some(*issued_times.get(&identity.uuid).ok_or_else(|| {
                anyhow!("captured request {} has no issuer timestamp", identity.uuid)
            })?)
        } else {
            None
        };
        Ok(())
    }
}

/// Source of the authoritative worker-built terminal record for a completed
/// online turn.
///
/// The online scheduled dispatcher records per-token facts in worker-local
/// observers, so the coordinator's adaptive window sampler never sees them
/// through the callback observer. This seam lets the per-phase adaptive record
/// processor pull each completed turn's finished [`RecordIngest`] and feed it to
/// the sampler through `WindowSampler::on_record`, exactly as the graph phase
/// runtime does. Backends whose dispatcher feeds the callback observer directly
/// (offline co-simulation) supply no source and keep the observer feed, so the
/// sampler is never double-fed.
pub(crate) trait AdaptiveTerminalRecordSource {
    /// Consume the finished record for `uuid`, if the worker produced one.
    fn take_terminal_record(&self, uuid: Uuid) -> Option<RecordIngest>;
}

impl AdaptiveTerminalRecordSource for RunCapture {
    fn take_terminal_record(&self, uuid: Uuid) -> Option<RecordIngest> {
        self.adaptive_records.borrow_mut().remove(&uuid)
    }
}

/// Feeds each completed online turn's finished worker record into the adaptive
/// window sampler.
///
/// This uses `graph_phase_runtime`'s `sampler.on_record(&record.ingest)` feed
/// for the scheduled online path, where token/usage/terminal facts are recorded
/// worker-locally and the coordinator's callback observer only sees arrivals.
/// It runs after normal measurement and credit return, keeping token
/// accumulation off the coordinator.
pub(crate) struct AdaptiveSamplerRecordProcessor {
    pub(crate) source: Rc<dyn AdaptiveTerminalRecordSource>,
    pub(crate) sampler: SharedWindowSampler,
}

#[async_trait(?Send)]
impl TurnRecordProcessor for AdaptiveSamplerRecordProcessor {
    async fn process(&self, credit: &IssuedCredit, _outcome: &TurnDispatchOutcome) -> Result<()> {
        if let Some(ingest) = self.source.take_terminal_record(credit.turn.uuid) {
            self.sampler.borrow_mut().on_record(&ingest);
        }
        Ok(())
    }
}

pub(crate) struct CapturePhaseProcessor {
    pub(crate) capture: Rc<RunCapture>,
    pub(crate) phase: MetricsPhase,
    pub(crate) identity: PhaseIdentity,
    pub(crate) has_credit_timestamp: bool,
    pub(crate) live_sink: Option<Rc<dyn LiveResultsSink>>,
    pub(crate) heartbeat: Option<Rc<HeartbeatLane>>,
}

#[async_trait(?Send)]
impl TurnRecordProcessor for CapturePhaseProcessor {
    async fn process(&self, credit: &IssuedCredit, outcome: &TurnDispatchOutcome) -> Result<()> {
        if self.capture.folds_records() {
            // Fold-and-drop mode (sketch or exact-fold): fold this turn's record into
            // the streaming accumulator and drop it, so peak memory stays
            // O(sketch)/O(scalars). A successful turn staged its record via
            // `record_live`; a failed or canceled turn never did (its `Err`/cancel path
            // skips `record_live`), so synthesize the record — matching the exact
            // path's finish-time fallback — to keep error counts and grouping correct.
            // Exact-fold also stamps the turn's absolute dispatch `request_index`
            // (assigned at `begin`) so its row lands at the retained-record path's
            // ordinal; sketch ignores it. The per-record live sink and cellular
            // heartbeat are not driven here: fold-and-drop retains no per-record data to
            // stream, and the exact-fold gate/sharded workers run with neither attached.
            let ingest = match self.capture.take_streaming_record(credit.turn.uuid) {
                Some(ingest) => ingest,
                None => self.capture.synthesize_streaming_fallback(credit, outcome),
            };
            let request_index = if self.capture.exact_fold {
                self.capture.take_fold_ordinal(credit.turn.uuid)
            } else {
                None
            };
            self.capture.fold_streaming(
                ingest,
                self.phase,
                &self.identity,
                self.has_credit_timestamp,
                request_index,
                credit,
            )?;
            return Ok(());
        }
        self.capture.label(
            credit,
            self.phase,
            &self.identity,
            self.has_credit_timestamp,
            outcome,
        );
        // The per-record clone is consumed once; feed both the Python live sink and
        // the cellular heartbeat lane from that single snapshot.
        if (self.live_sink.is_some() || self.heartbeat.is_some())
            && let Some(record) = self.capture.snapshot_live(credit)
        {
            if let Some(sink) = &self.live_sink {
                sink.emit_record(&record);
            }
            if let Some(heartbeat) = &self.heartbeat {
                heartbeat.observe_record(&record.ingest);
            }
        }
        Ok(())
    }
}

pub(crate) struct ConfiguredDispatcher {
    pub(crate) execution_backend: Rc<dyn RequestExecutor>,
    pub(crate) model: String,
    pub(crate) capture: Rc<RunCapture>,
}

#[async_trait(?Send)]
impl TurnDispatcher for ConfiguredDispatcher {
    fn inference_dimensions(&self, turn: &TurnToSend) -> InferenceDimensions {
        self.execution_backend.inference_dimensions(turn)
    }

    async fn dispatch_turn(
        &self,
        turn: TurnToSend,
        _observer: &dyn RequestObserver,
        on_first_token: &dyn Fn(i64),
    ) -> Result<TurnDispatchOutcome> {
        let uuid = turn.uuid;
        // Retain the conversation identity for `inputs.json` session grouping
        // before `turn` is consumed by request preparation below.
        let inputs_conversation_id = turn.conversation_id.clone();
        let inputs_turn_index = turn.turn_index;
        // `begin` runs on the coordinator thread before backend dispatch, so its
        // push order is the worker-count-independent global dispatch ordinal.
        // It returns the measured context the worker registers locally; the
        // runner's native-v2 report is then produced from the drained worker
        // records, not a single coordinator observer. The ScheduledRuntime's own
        // observer (`_observer`) is still computed and discarded by the runner.
        let context = self.capture.begin(&turn);
        let turn = PreparedTurn::from_turn(turn, &self.model);
        match self
            .execution_backend
            .execute_measured(turn, context, on_first_token)
            .await
        {
            Ok(MeasuredOutcome {
                result: collected,
                live_record,
            }) => {
                let outcome = collected.outcome;
                self.capture.record_input_payload(
                    &inputs_conversation_id,
                    inputs_turn_index,
                    &collected.request_payload,
                )?;
                self.capture.record_http_exchange(
                    uuid,
                    collected.request_payload,
                    collected.record,
                )?;
                self.capture.record_model_output(
                    uuid,
                    &outcome.response_text,
                    outcome.model_response.content.as_deref(),
                    outcome.model_response.reasoning.as_deref(),
                )?;
                if let Some(live_record) = live_record {
                    self.capture.record_live(uuid, live_record);
                }
                Ok(outcome)
            }
            // The worker (or, for a pre-dispatch failure, the coordinator
            // fallback at finish) owns finalizing the failed record; the
            // dispatcher only propagates the error.
            Err(error) => Err(error),
        }
    }

    async fn prewarm(&self, turn: TurnToSend) -> Result<()> {
        // Warm the execution backend (every worker) with the real prepared
        // request shape; the backend discards the round-trip and records
        // nothing, so timed issuance starts from a warmed transport.
        let turn = PreparedTurn::from_turn(turn, &self.model);
        self.execution_backend.prewarm(turn).await
    }
}

#[cfg(test)]
mod tests {
    use crate::clock::SimClock;

    use super::*;

    /// Distinct per-request facts for the `RunCapture::finish` join tests.
    struct RequestFacts {
        uuid: Uuid,
        arrival_ms: f64,
        token_times_ms: &'static [f64],
        prompt_tokens: u64,
        completion_tokens: u64,
        start_ns: i64,
        end_ns: i64,
    }

    /// Drive one request through a worker observer exactly as
    /// `TransportSink::dispatch_measured` does: register begin-known
    /// metadata (no `request_index` — the worker uses a dense-local arrival slot),
    /// arrival, admit, per-token arrivals, terminal, and the authoritative
    /// transport/usage response.
    fn drive_worker_request(observer: &NativeMetricsObserver, facts: &RequestFacts) {
        observer.register_metadata(facts.uuid, RequestMetricMetadata::default());
        observer.on_arrival(
            facts.uuid,
            facts.arrival_ms,
            facts.prompt_tokens as usize,
            8,
        );
        observer.on_admit(facts.uuid, facts.arrival_ms, 0);
        for &at in facts.token_times_ms {
            observer.on_token(facts.uuid, at);
        }
        observer.on_terminal(facts.uuid, ReplayTerminalStatus::Completed);
        observer.record_response(
            facts.uuid,
            NativeResponseMetadata {
                start_ns: Some(facts.start_ns),
                end_ns: Some(facts.end_ns),
                prompt_tokens: Some(facts.prompt_tokens),
                completion_tokens: Some(facts.completion_tokens),
                ..NativeResponseMetadata::default()
            },
        );
    }

    /// Register a dispatch identity + its coordinator label, exactly as `begin`
    /// and `label` do on the coordinator thread.
    fn register_identity(
        capture: &RunCapture,
        x_correlation_id: &str,
        session_num: u64,
        terminal: ReplayTerminalStatus,
        facts: &RequestFacts,
    ) {
        capture.identities.borrow_mut().push(CaptureIdentity {
            uuid: facts.uuid,
            x_correlation_id: x_correlation_id.to_string(),
            context: MeasuredContext {
                arrival_ms: facts.arrival_ms,
                input_length: facts.prompt_tokens as usize,
                requested_output_length: 8,
                metadata: RequestMetricMetadata::default(),
                wants_live_record: false,
                consume_record: false,
            },
        });
        capture.labels.borrow_mut().insert(
            facts.uuid,
            CaptureLabel {
                phase: MetricsPhase::Profiling,
                phase_index: None,
                phase_name: None,
                phase_kind: None,
                profiling_index: None,
                session_num,
                has_credit_timestamp: true,
                terminal,
                start_ns: facts.start_ns,
                end_ns: facts.end_ns,
            },
        );
    }

    fn facts() -> (RequestFacts, RequestFacts, RequestFacts) {
        (
            RequestFacts {
                uuid: Uuid::from_u128(0xA),
                arrival_ms: 1.0,
                token_times_ms: &[5.0, 8.0],
                prompt_tokens: 4,
                completion_tokens: 2,
                start_ns: 2_000_000,
                end_ns: 9_000_000,
            },
            RequestFacts {
                uuid: Uuid::from_u128(0xB),
                arrival_ms: 2.0,
                token_times_ms: &[6.0, 10.0, 14.0],
                prompt_tokens: 5,
                completion_tokens: 3,
                start_ns: 3_000_000,
                end_ns: 15_000_000,
            },
            RequestFacts {
                uuid: Uuid::from_u128(0xC),
                arrival_ms: 3.0,
                token_times_ms: &[7.0],
                prompt_tokens: 6,
                completion_tokens: 1,
                start_ns: 4_000_000,
                end_ns: 8_000_000,
            },
        )
    }

    /// inputs.json parity: the during-run capture path (fed in arbitrary
    /// dispatch order, with a recycled duplicate turn) and the up-front, dataset-ordered
    /// generation both funnel through the same `write_inputs_json`, so the two files are
    /// byte-identical — dedup per `(conversation_id, turn)` and the conversation-id sort
    /// are order-independent.
    #[test]
    fn inputs_json_up_front_matches_capture_regardless_of_dispatch_order() {
        let clock: Rc<dyn Clock> = Rc::new(SimClock::new());
        let capture = RunCapture::new(
            clock,
            0,
            MetricsConfig::default(),
            false,
            true, // inputs_enabled
            false,
            false,
            false,
        );
        // Distinct canonical bodies per (conversation, turn). "conv-b" is a two-turn
        // conversation; "conv-a" is single-turn.
        let body = |tag: &str| format!(r#"{{"model":"m","tag":"{tag}"}}"#).into_bytes();
        // Feed the during-run capture out of conversation order, with a recycled
        // duplicate (conv-b turn 0 dispatched twice, e.g. via --request-count recycling);
        // the second write must be ignored (first-write-wins dedup).
        capture
            .record_input_payload("conv-b", 0, &body("b0"))
            .unwrap();
        capture
            .record_input_payload("conv-a", 0, &body("a0"))
            .unwrap();
        capture
            .record_input_payload("conv-b", 1, &body("b1"))
            .unwrap();
        capture
            .record_input_payload("conv-b", 0, &body("b0-recycled"))
            .unwrap();
        let capture_sessions = capture.take_input_sessions();

        // The up-front generator emits sessions conversation-id-sorted, each with its
        // turns in order — build the equivalent list directly.
        let parse = |bytes: Vec<u8>| {
            serde_json::from_slice::<Box<serde_json::value::RawValue>>(&bytes).unwrap()
        };
        let up_front = vec![
            InputSession {
                session_id: "conv-a".into(),
                payloads: vec![parse(body("a0"))],
            },
            InputSession {
                session_id: "conv-b".into(),
                payloads: vec![parse(body("b0")), parse(body("b1"))],
            },
        ];

        let dir = tempfile::tempdir().unwrap();
        let capture_path = dir.path().join("inputs_capture.json");
        let up_front_path = dir.path().join("inputs_up_front.json");
        write_inputs_json(&capture_path, &capture_sessions).unwrap();
        write_inputs_json(&up_front_path, &up_front).unwrap();

        assert_eq!(
            std::fs::read(&capture_path).unwrap(),
            std::fs::read(&up_front_path).unwrap(),
            "up-front inputs.json must be byte-identical to the capture-based output"
        );
    }

    /// Worker records arrive per worker, not in global dispatch
    /// order. `RunCapture::finish` must key each record to its identity by uuid,
    /// emit rows in dispatch order, and stamp each record's `request_index` to its
    /// global dispatch ordinal so the downstream re-ingest is collision-free
    /// because a per-worker-local `request_index=Some(0)` collision would
    /// otherwise panic `insert_record_at`.
    #[test]
    fn run_capture_finish_stamps_global_index_and_joins_worker_records() {
        let clock: Rc<dyn Clock> = Rc::new(SimClock::new());
        let capture = RunCapture::new(
            clock.clone(),
            0,
            MetricsConfig::default(),
            false,
            false,
            false,
            false,
            false,
        );
        let (a, b, c) = facts();
        // Dispatch order A, B, C.
        register_identity(&capture, "corr-a", 0, ReplayTerminalStatus::Completed, &a);
        register_identity(&capture, "corr-b", 1, ReplayTerminalStatus::Completed, &b);
        register_identity(&capture, "corr-c", 2, ReplayTerminalStatus::Completed, &c);
        // Two workers: worker0 handled A then C (local slots 0, 1); worker1 handled
        // B (local slot 0). Concatenated drain order is [A, C, B] != dispatch order.
        let worker0 = NativeMetricsObserver::new(clock.clone(), 0, MetricsConfig::default());
        let worker1 = NativeMetricsObserver::new(clock.clone(), 0, MetricsConfig::default());
        drive_worker_request(&worker0, &a);
        drive_worker_request(&worker0, &c);
        drive_worker_request(&worker1, &b);
        let mut drained = worker0.finish_with_records().records;
        drained.extend(worker1.finish_with_records().records);
        assert_eq!(
            drained.iter().map(|(uuid, _)| *uuid).collect::<Vec<_>>(),
            vec![a.uuid, c.uuid, b.uuid],
        );
        // Both worker0 records carry local request_index Some(0)/Some(1) and
        // worker1's is Some(0): a raw re-ingest would collide.
        assert_eq!(drained[0].1.request_index, Some(0));
        assert_eq!(drained[2].1.request_index, Some(0));

        let issued: HashMap<Uuid, i64> = [(a.uuid, 111), (b.uuid, 222), (c.uuid, 333)]
            .into_iter()
            .collect();
        let captured = capture.finish(&issued, drained).unwrap();

        assert_eq!(
            captured.iter().map(|r| r.uuid).collect::<Vec<_>>(),
            vec![a.uuid, b.uuid, c.uuid],
        );
        assert_eq!(
            captured
                .iter()
                .map(|r| r.x_correlation_id.as_str())
                .collect::<Vec<_>>(),
            vec!["corr-a", "corr-b", "corr-c"],
        );
        // Global dispatch ordinal stamped dense 0..N-1 in dispatch order.
        assert_eq!(
            captured
                .iter()
                .map(|r| r.ingest.request_index)
                .collect::<Vec<_>>(),
            vec![Some(0), Some(1), Some(2)],
        );
        // uuid join: each row carries its own record; admit patched per uuid.
        for record in &captured {
            assert_eq!(record.ingest.correlation_id, record.uuid.to_string());
        }
        assert_eq!(captured[0].ingest.admit_ns, Some(111));
        assert_eq!(captured[1].ingest.admit_ns, Some(222));
        assert_eq!(captured[2].ingest.admit_ns, Some(333));
        // Re-ingest is collision-free (no insert_record_at panic) and counts all 3.
        let mut accumulator = MetricsAccumulator::with_config(MetricsConfig::default());
        for record in &captured {
            accumulator.process_record(&record.ingest);
        }
        let summary = accumulator.export_results(&ExportContext::phase(MetricsPhase::Profiling));
        assert_eq!(summary.finite_value(MetricTag::RequestCount), Some(3.0));
    }

    /// The same request set produces a byte-identical re-ingested report whether it
    /// drains from one worker or is split across two. `finish` stamps the
    /// global dispatch ordinal in both cases, so the IEEE-754 fold order is
    /// identical and no worker-count reorder occurs on the runner path.
    #[test]
    fn run_capture_finish_worker_split_matches_single_worker_byte_for_byte() {
        let build = |split: bool| -> (Vec<u8>, Option<f64>) {
            let clock: Rc<dyn Clock> = Rc::new(SimClock::new());
            let config = MetricsConfig::default();
            let capture = RunCapture::new(
                clock.clone(),
                0,
                config.clone(),
                false,
                false,
                false,
                false,
                false,
            );
            let (a, b, c) = facts();
            register_identity(&capture, "corr-a", 0, ReplayTerminalStatus::Completed, &a);
            register_identity(&capture, "corr-b", 1, ReplayTerminalStatus::Completed, &b);
            register_identity(&capture, "corr-c", 2, ReplayTerminalStatus::Completed, &c);
            let drained = if split {
                let worker0 = NativeMetricsObserver::new(clock.clone(), 0, config.clone());
                let worker1 = NativeMetricsObserver::new(clock.clone(), 0, config.clone());
                drive_worker_request(&worker0, &a);
                drive_worker_request(&worker0, &c);
                drive_worker_request(&worker1, &b);
                let mut drained = worker0.finish_with_records().records;
                drained.extend(worker1.finish_with_records().records);
                drained
            } else {
                let worker = NativeMetricsObserver::new(clock.clone(), 0, config.clone());
                drive_worker_request(&worker, &a);
                drive_worker_request(&worker, &b);
                drive_worker_request(&worker, &c);
                worker.finish_with_records().records
            };
            let issued: HashMap<Uuid, i64> = [
                (a.uuid, 1_500_000),
                (b.uuid, 2_500_000),
                (c.uuid, 3_500_000),
            ]
            .into_iter()
            .collect();
            let captured = capture.finish(&issued, drained).unwrap();
            let mut accumulator = MetricsAccumulator::with_config(config);
            for record in &captured {
                accumulator.process_record(&record.ingest);
            }
            let summary =
                accumulator.export_results(&ExportContext::phase(MetricsPhase::Profiling));
            (
                serde_json::to_vec(&summary).unwrap(),
                summary.finite_value(MetricTag::RequestCount),
            )
        };
        let (single_bytes, single_count) = build(false);
        let (split_bytes, split_count) = build(true);
        assert_eq!(single_count, Some(3.0));
        assert_eq!(split_count, Some(3.0));
        assert_eq!(
            single_bytes, split_bytes,
            "worker-split drain must re-ingest byte-identically to a single worker",
        );
    }

    /// The live-results sink must read a non-consuming clone. The
    /// worker returns `snapshot_record` (not `drain_terminal_record`), so the
    /// authoritative record stays in the worker observer and the end-of-run drain
    /// still counts every live-emitted request — a `--live` run cannot undercount.
    #[test]
    fn live_record_snapshot_does_not_consume_the_drain_record() {
        let clock: Rc<dyn Clock> = Rc::new(SimClock::new());
        let observer = NativeMetricsObserver::new(clock, 0, MetricsConfig::default());
        let (a, _b, _c) = facts();
        drive_worker_request(&observer, &a);
        // The live sink emits from this non-consuming clone.
        let live = observer.snapshot_record(a.uuid, 0);
        assert!(live.is_some(), "a terminal request yields a live snapshot");
        // The authoritative record is still present for the end-of-run drain.
        let drained = observer.finish_with_records().records;
        assert_eq!(drained.len(), 1);
        assert_eq!(drained[0].0, a.uuid);
    }

    /// An identity that fails before any worker observer registers it has no drained
    /// record. `finish` must synthesize an errored
    /// fallback record so `RequestCount`/`ErrorRequestCount` stay exact and the run
    /// does not abort fail-closed on the missing lookup.
    #[test]
    fn run_capture_finish_synthesizes_fallback_for_pre_worker_failures() {
        let clock: Rc<dyn Clock> = Rc::new(SimClock::new());
        let capture = RunCapture::new(
            clock.clone(),
            0,
            MetricsConfig::default(),
            false,
            false,
            false,
            false,
            false,
        );
        let (a, b, _c) = facts();
        register_identity(&capture, "corr-a", 0, ReplayTerminalStatus::Completed, &a);
        // B is dispatched (identity + Failed label) but never reaches a worker.
        register_identity(&capture, "corr-b", 1, ReplayTerminalStatus::Failed, &b);
        let worker = NativeMetricsObserver::new(clock.clone(), 0, MetricsConfig::default());
        drive_worker_request(&worker, &a);
        let drained = worker.finish_with_records().records;
        assert_eq!(drained.len(), 1);

        let issued: HashMap<Uuid, i64> = [(a.uuid, 111), (b.uuid, 222)].into_iter().collect();
        let captured = capture.finish(&issued, drained).unwrap();

        assert_eq!(
            captured.iter().map(|r| r.uuid).collect::<Vec<_>>(),
            vec![a.uuid, b.uuid],
        );
        assert_eq!(
            captured
                .iter()
                .map(|r| r.ingest.request_index)
                .collect::<Vec<_>>(),
            vec![Some(0), Some(1)],
        );
        assert!(!captured[0].ingest.errored);
        assert!(
            captured[1].ingest.errored,
            "the pre-worker failure is errored"
        );

        let mut accumulator = MetricsAccumulator::with_config(MetricsConfig::default());
        for record in &captured {
            accumulator.process_record(&record.ingest);
        }
        let summary = accumulator.export_results(&ExportContext::phase(MetricsPhase::Profiling));
        // RequestCount counts successes only; the errored fallback lands in
        // ErrorRequestCount, so the total CompletedRequestCount is 2.
        assert_eq!(summary.finite_value(MetricTag::RequestCount), Some(1.0));
        assert_eq!(
            summary.finite_value(MetricTag::ErrorRequestCount),
            Some(1.0)
        );
        assert_eq!(
            summary.finite_value(MetricTag::CompletedRequestCount),
            Some(2.0)
        );
    }

    /// Folding each completed record into the exact accumulator in
    /// completion order, stamping the absolute dispatch `request_index` — and merging
    /// that accumulator into the report yields byte-identical exported results to the
    /// retained-record path's dispatch-order re-ingest, for BOTH the profiling and warmup
    /// windows and including an errored record's accounting. This is the core contract:
    /// exact-fold keeps exact NaN-sparse columns (not the sketch approximation), so the
    /// mid-run fold-and-drop is invisible in the summary.
    #[test]
    fn exact_fold_matches_compatibility_retain_byte_for_byte() {
        let clock: Rc<dyn Clock> = Rc::new(SimClock::new());
        let config = MetricsConfig::default();

        // Four realistic worker-drained records; array index i is dispatch ordinal i.
        let source_facts = [
            RequestFacts {
                uuid: Uuid::from_u128(0x11),
                arrival_ms: 1.0,
                token_times_ms: &[5.0, 8.0],
                prompt_tokens: 4,
                completion_tokens: 2,
                start_ns: 2_000_000,
                end_ns: 9_000_000,
            },
            RequestFacts {
                uuid: Uuid::from_u128(0x22),
                arrival_ms: 2.0,
                token_times_ms: &[6.0, 10.0, 14.0],
                prompt_tokens: 5,
                completion_tokens: 3,
                start_ns: 3_000_000,
                end_ns: 15_000_000,
            },
            RequestFacts {
                uuid: Uuid::from_u128(0x33),
                arrival_ms: 3.0,
                token_times_ms: &[7.0, 9.0],
                prompt_tokens: 6,
                completion_tokens: 2,
                start_ns: 4_000_000,
                end_ns: 12_000_000,
            },
            RequestFacts {
                uuid: Uuid::from_u128(0x44),
                arrival_ms: 4.0,
                token_times_ms: &[8.0],
                prompt_tokens: 7,
                completion_tokens: 1,
                start_ns: 5_000_000,
                end_ns: 8_000_000,
            },
        ];
        // Per-record coordinator-owned facts: phase (mix of warmup + profiling),
        // session number, admit ns, and whether the record errored (record 3).
        let phases = [
            MetricsPhase::Profiling,
            MetricsPhase::Warmup,
            MetricsPhase::Profiling,
            MetricsPhase::Profiling,
        ];
        let admits = [1_500_000i64, 2_500_000, 3_500_000, 4_500_000];
        let is_errored = [false, false, false, true];

        // Build the drained ingests fresh (dispatch order), applying the errored flag.
        let build_records = || -> Vec<RecordIngest> {
            source_facts
                .iter()
                .enumerate()
                .map(|(i, facts)| {
                    let observer = NativeMetricsObserver::new(clock.clone(), 0, config.clone());
                    drive_worker_request(&observer, facts);
                    let mut ingest = observer
                        .finish_with_records()
                        .records
                        .into_iter()
                        .next()
                        .unwrap()
                        .1;
                    ingest.errored = is_errored[i];
                    ingest
                })
                .collect()
        };

        // Retained-record reference: patch coordinator-owned fields and process them
        // in dispatch order.
        let reference_summary = |phase: MetricsPhase| -> Vec<u8> {
            let mut accumulator = MetricsAccumulator::with_config(config.clone());
            for (i, mut ingest) in build_records().into_iter().enumerate() {
                ingest.phase = phases[i];
                ingest.session_num = i as u64;
                ingest.admit_ns = Some(admits[i]);
                ingest.request_index = Some(i);
                accumulator.process_record(&ingest);
            }
            serde_json::to_vec(&accumulator.export_results(&ExportContext::phase(phase))).unwrap()
        };

        // Fold each record in reverse completion order to validate order-independent
        // absolute-slot placement in the capture's exact
        // accumulator, then merge it into a fresh report accumulator.
        let subject_summary = |phase: MetricsPhase| -> Vec<u8> {
            let capture = RunCapture::new(
                clock.clone(),
                0,
                config.clone(),
                false,
                false,
                false,
                false,
                true,
            );
            assert!(
                capture.exact_fold && !capture.metrics_only,
                "exact-fold keeps EXACT storage, not sketch"
            );
            let records = build_records();
            for i in (0..records.len()).rev() {
                capture
                    .fold_record(
                        records[i].clone(),
                        source_facts[i].uuid,
                        "corr",
                        phases[i],
                        None,
                        None,
                        None,
                        None,
                        i as u64,
                        Some(admits[i]),
                        Some(i),
                    )
                    .unwrap();
            }
            let (streamed, errored_records) = capture.take_streamed();
            assert_eq!(
                errored_records.len(),
                1,
                "only the errored record is retained; the rest are dropped"
            );
            let mut accumulator = MetricsAccumulator::with_config(config.clone());
            accumulator.merge(&streamed).unwrap();
            serde_json::to_vec(&accumulator.export_results(&ExportContext::phase(phase))).unwrap()
        };

        assert_eq!(
            reference_summary(MetricsPhase::Profiling),
            subject_summary(MetricsPhase::Profiling),
            "profiling window must be byte-identical to the retain path",
        );
        assert_eq!(
            reference_summary(MetricsPhase::Warmup),
            subject_summary(MetricsPhase::Warmup),
            "warmup window must be byte-identical to the retain path",
        );
    }

    /// The exact-fold capture folds each profiling record's per-record OTLP
    /// histogram at completion (`with_otel` + `fold_record`), and `take_otel` yields
    /// the byte-identical accumulator the retain path builds by looping the retained
    /// records post-run — for the same record sequence. Warmup records never
    /// contribute, matching the post-run loop's `phase == Profiling` filter.
    #[test]
    fn fold_record_folds_otel_matching_post_run_loop() {
        let clock: Rc<dyn Clock> = Rc::new(SimClock::new());
        let config = MetricsConfig::default();

        // (uuid, isl, osl, end_ns, phase, errored) — a mix of profiling successes, one
        // errored profiling record, and one warmup record the OTLP fold must ignore.
        let make = |isl: u64, osl: u64, end_ns: i64, phase: MetricsPhase| -> RecordIngest {
            let mut ingest = RecordIngest::minimal(1_000_000, end_ns, phase);
            ingest.first_token_ns = Some(3_000_000);
            ingest.token_arrival_ns = vec![3_000_000, 5_000_000, end_ns];
            ingest.tokens = crate::metrics_core::TokenCounts {
                input: Some(isl),
                output: Some(osl),
                requested_output: Some(osl),
                ..Default::default()
            };
            ingest
        };
        let specs: Vec<(Uuid, RecordIngest, bool)> = vec![
            (
                Uuid::from_u128(0x1),
                make(8, 3, 11_000_000, MetricsPhase::Profiling),
                false,
            ),
            (
                Uuid::from_u128(0x2),
                make(16, 5, 21_000_000, MetricsPhase::Profiling),
                false,
            ),
            (
                Uuid::from_u128(0x3),
                make(64, 1, 4_000_000, MetricsPhase::Profiling),
                true,
            ),
            (
                Uuid::from_u128(0x4),
                make(8, 3, 11_000_000, MetricsPhase::Warmup),
                false,
            ),
        ];

        // Exact-fold path: fold each record at completion with native OTLP enabled.
        let capture = RunCapture::new(
            clock.clone(),
            0,
            config.clone(),
            false,
            false,
            false,
            false,
            true,
        )
        .with_otel(true);
        for (i, (uuid, ingest, errored)) in specs.iter().enumerate() {
            let mut ingest = ingest.clone();
            ingest.errored = *errored;
            let phase = ingest.phase;
            capture
                .fold_record(
                    ingest,
                    *uuid,
                    "corr",
                    phase,
                    None,
                    None,
                    None,
                    None,
                    i as u64,
                    None,
                    Some(i),
                )
                .unwrap();
        }
        let folded = capture.take_otel().expect("otel enabled");

        // Retain path: build the equivalent stamped records and fold the profiling
        // subset via the post-run loop, in the same order.
        let mut post_run = OtelRecordAccumulator::new();
        for (i, (uuid, ingest, errored)) in specs.iter().enumerate() {
            let mut ingest = ingest.clone();
            ingest.errored = *errored;
            ingest.session_num = i as u64;
            ingest.request_index = Some(i);
            if ingest.phase == MetricsPhase::Profiling {
                let captured = CapturedRecord {
                    uuid: *uuid,
                    x_correlation_id: "corr".into(),
                    output: CapturedModelOutput::default(),
                    raw: None,
                    ingest,
                };
                observe_otel_record(&mut post_run, &captured, &config);
            }
        }

        assert!(
            !folded.is_empty(),
            "profiling records populate the histograms"
        );
        assert_eq!(
            folded, post_run,
            "fold-at-completion OTLP must equal the post-run-loop OTLP for the same sequence"
        );
    }

    /// Exact-fold sets the fold-and-drop flags (so the worker consumes each record out
    /// of its observer) and assigns dense `0..N` dispatch ordinals at begin, consumed
    /// once at completion. A plain exact (retain) capture folds nothing.
    #[test]
    fn exact_fold_flags_and_dense_dispatch_ordinals() {
        let clock: Rc<dyn Clock> = Rc::new(SimClock::new());
        let config = MetricsConfig::default();
        let capture = RunCapture::new(
            clock.clone(),
            0,
            config.clone(),
            false,
            false,
            false,
            false,
            true,
        );
        assert!(
            capture.folds_records(),
            "exact-fold is a fold-and-drop mode"
        );
        assert!(
            capture.wants_live_record(),
            "the worker must return each record so the fold can consume it"
        );
        let a = Uuid::from_u128(0xA1);
        let b = Uuid::from_u128(0xB2);
        let c = Uuid::from_u128(0xC3);
        assert_eq!(capture.assign_fold_ordinal(a), 0);
        assert_eq!(capture.assign_fold_ordinal(b), 1);
        assert_eq!(capture.assign_fold_ordinal(c), 2);
        assert_eq!(capture.take_fold_ordinal(b), Some(1));
        assert_eq!(capture.take_fold_ordinal(b), None, "consumed exactly once");
        assert_eq!(capture.take_fold_ordinal(a), Some(0));
        assert_eq!(capture.take_fold_ordinal(c), Some(2));

        // The default exact (retain) capture folds nothing and needs no live record.
        let retain = RunCapture::new(clock, 0, config, false, false, false, false, false);
        assert!(!retain.folds_records());
        assert!(!retain.wants_live_record());
    }
}
