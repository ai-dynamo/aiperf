// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded fetch-to-native-to-archive attempt pipeline.
//!
//! Decode jobs run in a shared explicitly bounded CPU pool. Their result
//! returns to the source LocalSet, where native delivery occurs synchronously
//! before archive enqueue observation. The sole archive owner remains an
//! injected trait and therefore retains sequencing, projection, and durability
//! authority.

use std::collections::BTreeSet;
use std::fmt::{self, Debug, Display, Formatter};
use std::future::Future;
use std::pin::Pin;
use std::rc::Rc;
use std::sync::Arc;

use aiperf_clock::Clock;
use aiperf_prometheus::Exposition;
use aiperf_telemetry_archive::{
    AdmissionRejection, ArchiveAttemptProjectionContextV1, AttemptDecoder, BoundaryReference,
    DecodeLimits, DecodedAttempt, DriverConsumerError, FetchedAttempt, LossKindV1, LossReasonV1,
    MissedCadenceRange, NativeEntityDecoder, NoopNativeEntityDecoder, PrometheusAttemptDecoder,
    ScrapeReasonV1, TelemetryAttemptConsumer, TelemetryAttemptDisposition,
    TelemetryAttemptEnvelope,
};
use async_trait::async_trait;
use tokio::sync::Semaphore;

/// Shared bounded CPU decode capacity across physical sources.
#[derive(Clone)]
pub struct BoundedTelemetryDecodePool {
    permits: Arc<Semaphore>,
    capacity: usize,
}

impl BoundedTelemetryDecodePool {
    /// Construct a positive-capacity pool before any source task starts.
    pub fn new(capacity: usize) -> Result<Self, DecodePoolError> {
        if capacity == 0 {
            return Err(DecodePoolError::ZeroCapacity);
        }
        Ok(Self {
            permits: Arc::new(Semaphore::new(capacity)),
            capacity,
        })
    }

    async fn decode(
        &self,
        decoder: Arc<dyn AttemptDecoder<Exposition, ()>>,
        fetched: FetchedAttempt,
        limits: DecodeLimits,
    ) -> Result<DecodedAttempt<Exposition, ()>, DecodePoolError> {
        let permit = self
            .permits
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| DecodePoolError::Closed)?;
        tokio::task::spawn_blocking(move || {
            let decoded = decoder.decode(fetched, &limits);
            drop(permit);
            decoded
        })
        .await
        .map_err(|error| {
            DecodePoolError::Task(if error.is_cancelled() {
                "telemetry decode task was cancelled".to_owned()
            } else {
                format!("telemetry decode task panicked: {error}")
            })
        })
    }

    /// Frozen maximum simultaneously executing decode jobs.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    #[cfg(test)]
    fn available_permits(&self) -> usize {
        self.permits.available_permits()
    }
}

impl Debug for BoundedTelemetryDecodePool {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BoundedTelemetryDecodePool")
            .field("capacity", &self.capacity)
            .field("available", &self.permits.available_permits())
            .finish()
    }
}

/// LocalSet Clock-stamped terminal decode result handed to the archive owner.
#[derive(Debug)]
pub struct ArchiveAttemptObservation {
    /// All-outcome strict/native decode result.
    pub decoded: DecodedAttempt<Exposition, ()>,
    /// Clock instant after bounded decode became terminal.
    pub parse_done_ns: i64,
    /// Clock instant after native delivery and immediately before archive handoff.
    pub archive_enqueue_ns: i64,
    /// Driver-owned cadence or exact sealed-boundary projection context.
    pub projection_context: ArchiveAttemptProjectionContextV1,
}

/// Local future resolving one admitted attached attempt's owner terminalization.
pub type AttachedAttemptTerminalFuture =
    Pin<Box<dyn Future<Output = Result<TelemetryAttemptDisposition, ArchiveOwnerError>> + 'static>>;

/// Immediate result of nonblocking attached owner admission.
pub struct AttachedAttemptAdmission {
    /// Boundary attempts retain a terminal future; continuous attempts do not
    /// make their source driver wait for projection, WAL, or receipt work.
    pub boundary_terminal: Option<AttachedAttemptTerminalFuture>,
}

impl Debug for AttachedAttemptAdmission {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AttachedAttemptAdmission")
            .field("boundary_terminal", &self.boundary_terminal.is_some())
            .finish()
    }
}

/// Clock-stamped issued-work loss recorded without using the data queue.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArchiveIssuedLossObservation {
    /// Stable physical source identity.
    pub source_id: String,
    /// Per-source event sequence that native delivery already observed.
    pub source_record_seq: u64,
    /// Per-source request sequence when network work began.
    pub request_attempt_seq: Option<u64>,
    /// Closed semantic loss class.
    pub loss_kind: LossKindV1,
    /// Closed reason paired with `loss_kind`.
    pub reason: LossReasonV1,
    /// LocalSet Clock instant when loss became observable.
    pub observed_ns: i64,
    /// Exact boundary joins retained on source-scoped loss.
    pub boundary_refs: Vec<BoundaryReference>,
}

/// Compact missed cadence fact handed to the same archive owner.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArchiveMissedObservation {
    /// Stable physical source identity.
    pub source_id: String,
    /// Exact inclusive missed range.
    pub missed: MissedCadenceRange,
    /// LocalSet Clock instant when the gap became observable.
    pub observed_ns: i64,
}

/// Driver-owned context supplied to authoritative native projection.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeAttemptContext {
    /// Continuous cadence or forced boundary reason.
    pub reason: ScrapeReasonV1,
    /// Active phase membership at the source snapshot instant.
    pub active_phase_ids: BTreeSet<String>,
    /// Exact boundary joins independent of active phase membership.
    pub boundary_refs: Vec<BoundaryReference>,
}

/// Synchronous native projection/accumulator hook.
pub trait NativeAttemptObserver: Debug {
    /// Deliver a decoded native entity before archive admission or persistence.
    fn observe(&self, attempt: &DecodedAttempt<Exposition, ()>) -> Result<(), NativeObserverError>;

    /// Deliver with snapshot membership and boundary context.
    ///
    /// Existing standalone observers remain source compatible. Attached native
    /// accumulators override this method so one decoded physical attempt feeds
    /// every active phase before archive admission is attempted.
    fn observe_with_context(
        &self,
        attempt: &DecodedAttempt<Exposition, ()>,
        _context: &NativeAttemptContext,
    ) -> Result<(), NativeObserverError> {
        self.observe(attempt)
    }
}

/// Standalone-watch observer with deliberately no native metric projection.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoopNativeAttemptObserver;

impl NativeAttemptObserver for NoopNativeAttemptObserver {
    fn observe(
        &self,
        _attempt: &DecodedAttempt<Exposition, ()>,
    ) -> Result<(), NativeObserverError> {
        Ok(())
    }
}

/// Single-owner sequencing/durability seam after LocalSet observation.
#[async_trait(?Send)]
pub trait ArchiveAttemptOwner: Debug {
    /// Accept one all-outcome decoded attempt in source FIFO order.
    async fn observe_attempt(
        &self,
        observation: ArchiveAttemptObservation,
    ) -> Result<(), ArchiveOwnerError>;

    /// Accept one compact cadence-loss range.
    async fn observe_missed(
        &self,
        observation: ArchiveMissedObservation,
    ) -> Result<(), ArchiveOwnerError>;
}

/// Nonblocking attached owner seam used only after synchronous native delivery.
pub trait AttachedArchiveAttemptOwner: Debug {
    /// Try to admit one decoded projection using already reserved data capacity.
    ///
    /// The call never waits for owner scheduling, projection, WAL, or receipts.
    /// Boundary admission returns a future because the source-cardinal phase
    /// barrier must learn the exact attempt-or-loss terminalization.
    fn try_observe_attempt(
        &self,
        observation: ArchiveAttemptObservation,
    ) -> Result<AttachedAttemptAdmission, AdmissionRejection>;

    /// Record issued-work loss through capacity independent of the data queue.
    fn record_visible_loss(
        &self,
        observation: ArchiveIssuedLossObservation,
    ) -> Result<(), ArchiveOwnerError>;

    /// Record compact cadence loss without making the driver wait on WAL work.
    fn record_missed(&self, observation: ArchiveMissedObservation)
    -> Result<(), ArchiveOwnerError>;
}

enum ArchivePipelineOwner {
    Standalone(Rc<dyn ArchiveAttemptOwner>),
    Attached(Rc<dyn AttachedArchiveAttemptOwner>),
}

impl Debug for ArchivePipelineOwner {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Standalone(owner) => formatter.debug_tuple("Standalone").field(owner).finish(),
            Self::Attached(owner) => formatter.debug_tuple("Attached").field(owner).finish(),
        }
    }
}

/// Prepared source-local consumer sharing global decode capacity and one owner.
pub struct PrometheusAttemptPipeline {
    clock: Rc<dyn Clock>,
    decoder: Arc<dyn AttemptDecoder<Exposition, ()>>,
    limits: DecodeLimits,
    decode_pool: BoundedTelemetryDecodePool,
    native: Rc<dyn NativeAttemptObserver>,
    owner: ArchivePipelineOwner,
}

impl PrometheusAttemptPipeline {
    /// Compose the strict parser, native hook, and archive owner.
    pub fn new(
        clock: Rc<dyn Clock>,
        decoder: Arc<dyn AttemptDecoder<Exposition, ()>>,
        limits: DecodeLimits,
        decode_pool: BoundedTelemetryDecodePool,
        native: Rc<dyn NativeAttemptObserver>,
        owner: Rc<dyn ArchiveAttemptOwner>,
    ) -> Result<Self, PipelinePrepareError> {
        limits
            .validate()
            .map_err(|error| PipelinePrepareError::DecodeLimits(error.to_string()))?;
        Ok(Self {
            clock,
            decoder,
            limits,
            decode_pool,
            native,
            owner: ArchivePipelineOwner::Standalone(owner),
        })
    }

    /// Compose one attached native-first pipeline over a nonblocking owner.
    pub fn new_attached(
        clock: Rc<dyn Clock>,
        decoder: Arc<dyn AttemptDecoder<Exposition, ()>>,
        limits: DecodeLimits,
        decode_pool: BoundedTelemetryDecodePool,
        native: Rc<dyn NativeAttemptObserver>,
        owner: Rc<dyn AttachedArchiveAttemptOwner>,
    ) -> Result<Self, PipelinePrepareError> {
        limits
            .validate()
            .map_err(|error| PipelinePrepareError::DecodeLimits(error.to_string()))?;
        Ok(Self {
            clock,
            decoder,
            limits,
            decode_pool,
            native,
            owner: ArchivePipelineOwner::Attached(owner),
        })
    }

    /// Compose the stock strict exposition decoder with no native entity.
    pub fn strict_standalone(
        clock: Rc<dyn Clock>,
        parser: Arc<dyn aiperf_prometheus::ExpositionParser>,
        limits: DecodeLimits,
        decode_pool: BoundedTelemetryDecodePool,
        owner: Rc<dyn ArchiveAttemptOwner>,
    ) -> Result<Self, PipelinePrepareError> {
        let native: Arc<dyn NativeEntityDecoder<()>> = Arc::new(NoopNativeEntityDecoder);
        let decoder: Arc<dyn AttemptDecoder<Exposition, ()>> =
            Arc::new(PrometheusAttemptDecoder::new(parser, native));
        Self::new(
            clock,
            decoder,
            limits,
            decode_pool,
            Rc::new(NoopNativeAttemptObserver),
            owner,
        )
    }

    async fn observe_envelope(
        &self,
        envelope: TelemetryAttemptEnvelope,
    ) -> Result<TelemetryAttemptDisposition, DriverConsumerError> {
        let native_context = NativeAttemptContext {
            reason: envelope.reason,
            active_phase_ids: envelope.active_phase_ids,
            boundary_refs: envelope.boundary_refs.clone(),
        };
        let decoded = self
            .decode_pool
            .decode(self.decoder.clone(), envelope.attempt, self.limits.clone())
            .await
            .map_err(consumer_error)?;
        let parse_done_ns = self.clock.now_ns();
        self.native
            .observe_with_context(&decoded, &native_context)
            .map_err(consumer_error)?;
        let archive_enqueue_ns = self.clock.now_ns();
        let mut loss_identity = ArchiveIssuedLossObservation {
            source_id: decoded.facts.source_id.clone(),
            source_record_seq: decoded.facts.source_record_seq,
            request_attempt_seq: decoded.facts.request_attempt_seq,
            loss_kind: LossKindV1::ArchiveRejected,
            reason: LossReasonV1::ArchiveAdmissionRejected,
            observed_ns: archive_enqueue_ns,
            boundary_refs: native_context.boundary_refs.clone(),
        };
        let observation = ArchiveAttemptObservation {
            decoded,
            parse_done_ns,
            archive_enqueue_ns,
            projection_context: ArchiveAttemptProjectionContextV1 {
                reason: native_context.reason,
                boundary_refs: native_context.boundary_refs,
            },
        };

        match &self.owner {
            ArchivePipelineOwner::Standalone(owner) => {
                owner
                    .observe_attempt(observation)
                    .await
                    .map_err(consumer_error)?;
                Ok(TelemetryAttemptDisposition::Attempt)
            }
            ArchivePipelineOwner::Attached(owner) => {
                let admission = match owner.try_observe_attempt(observation) {
                    Ok(admission) => admission,
                    Err(rejection) => {
                        let (kind, reason) = match rejection {
                            AdmissionRejection::Closed => {
                                (LossKindV1::WriterFailed, LossReasonV1::WriterError)
                            }
                            AdmissionRejection::Capacity | AdmissionRejection::ProtectedReserve => {
                                (
                                    LossKindV1::ArchiveRejected,
                                    LossReasonV1::ArchiveAdmissionRejected,
                                )
                            }
                        };
                        loss_identity.loss_kind = kind;
                        loss_identity.reason = reason;
                        loss_identity.observed_ns = self.clock.now_ns();
                        owner
                            .record_visible_loss(loss_identity)
                            .map_err(consumer_error)?;
                        return Ok(TelemetryAttemptDisposition::Loss { kind, reason });
                    }
                };
                if native_context.reason == ScrapeReasonV1::Boundary {
                    let terminal = admission.boundary_terminal.ok_or_else(|| {
                        consumer_error(ArchiveOwnerError {
                            message: "attached boundary admission omitted terminal future"
                                .to_owned(),
                        })
                    })?;
                    terminal.await.map_err(consumer_error)
                } else if admission.boundary_terminal.is_some() {
                    Err(consumer_error(ArchiveOwnerError {
                        message: "continuous attached admission returned a boundary future"
                            .to_owned(),
                    }))
                } else {
                    Ok(TelemetryAttemptDisposition::Attempt)
                }
            }
        }
    }
}

impl Debug for PrometheusAttemptPipeline {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PrometheusAttemptPipeline")
            .field("virtual_clock", &self.clock.is_virtual())
            .field("decoder", &self.decoder)
            .field("limits", &self.limits)
            .field("decode_pool", &self.decode_pool)
            .field("native", &self.native)
            .field("owner", &self.owner)
            .finish()
    }
}

#[async_trait(?Send)]
impl TelemetryAttemptConsumer for PrometheusAttemptPipeline {
    async fn observe_attempt(&self, attempt: FetchedAttempt) -> Result<(), DriverConsumerError> {
        self.observe_envelope(TelemetryAttemptEnvelope {
            attempt,
            reason: ScrapeReasonV1::Continuous,
            boundary_refs: Vec::new(),
            active_phase_ids: BTreeSet::new(),
        })
        .await
        .map(|_| ())
    }

    async fn observe_attempt_envelope(
        &self,
        envelope: TelemetryAttemptEnvelope,
    ) -> Result<TelemetryAttemptDisposition, DriverConsumerError> {
        self.observe_envelope(envelope).await
    }

    async fn observe_missed(
        &self,
        source_id: &str,
        missed: MissedCadenceRange,
    ) -> Result<(), DriverConsumerError> {
        let observation = ArchiveMissedObservation {
            source_id: source_id.to_owned(),
            missed,
            observed_ns: self.clock.now_ns(),
        };
        match &self.owner {
            ArchivePipelineOwner::Standalone(owner) => owner
                .observe_missed(observation)
                .await
                .map_err(consumer_error),
            ArchivePipelineOwner::Attached(owner) => {
                owner.record_missed(observation).map_err(consumer_error)
            }
        }
    }
}

fn consumer_error(error: impl Display) -> DriverConsumerError {
    DriverConsumerError {
        message: error.to_string(),
    }
}

/// Shared decode-pool failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DecodePoolError {
    /// Capacity zero cannot guarantee accepted attempt progress.
    ZeroCapacity,
    /// Pool was closed before a reserved job began.
    Closed,
    /// Blocking task was cancelled or panicked.
    Task(String),
}

impl Display for DecodePoolError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroCapacity => formatter.write_str("telemetry decode capacity must be positive"),
            Self::Closed => {
                formatter.write_str("telemetry decode pool closed before job admission")
            }
            Self::Task(message) => formatter.write_str(message),
        }
    }
}

impl std::error::Error for DecodePoolError {}

/// Static pipeline composition failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PipelinePrepareError {
    /// Strict decode limits are zero or internally inconsistent.
    DecodeLimits(String),
}

impl Display for PipelinePrepareError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::DecodeLimits(message) => {
                write!(formatter, "invalid telemetry decode limits: {message}")
            }
        }
    }
}

impl std::error::Error for PipelinePrepareError {}

/// Native delivery failure before archive admission.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeObserverError {
    /// Bounded redaction-safe detail.
    pub message: String,
}

impl Display for NativeObserverError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for NativeObserverError {}

/// Sole archive owner rejected one observation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArchiveOwnerError {
    /// Bounded redaction-safe detail.
    pub message: String,
}

impl Display for ArchiveOwnerError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for ArchiveOwnerError {}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use aiperf_clock::SimClock;
    use aiperf_prometheus::StrictExpositionParser;
    use aiperf_telemetry_archive::{FetchDisposition, SourceOutcome};
    use bytes::Bytes;

    use super::*;

    #[derive(Debug)]
    struct OrderingNative {
        events: Rc<RefCell<Vec<&'static str>>>,
    }

    impl NativeAttemptObserver for OrderingNative {
        fn observe(
            &self,
            attempt: &DecodedAttempt<Exposition, ()>,
        ) -> Result<(), NativeObserverError> {
            assert_eq!(attempt.facts.outcome, SourceOutcome::Success);
            self.events.borrow_mut().push("native");
            Ok(())
        }
    }

    #[derive(Debug)]
    struct CountingDecoder {
        calls: Arc<AtomicUsize>,
        inner: PrometheusAttemptDecoder<()>,
    }

    impl AttemptDecoder<Exposition, ()> for CountingDecoder {
        fn decode(
            &self,
            fetched: FetchedAttempt,
            limits: &DecodeLimits,
        ) -> DecodedAttempt<Exposition, ()> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.inner.decode(fetched, limits)
        }
    }

    #[derive(Debug)]
    struct ContextNative {
        events: Rc<RefCell<Vec<&'static str>>>,
        contexts: RefCell<Vec<NativeAttemptContext>>,
    }

    impl NativeAttemptObserver for ContextNative {
        fn observe(
            &self,
            _attempt: &DecodedAttempt<Exposition, ()>,
        ) -> Result<(), NativeObserverError> {
            Err(NativeObserverError {
                message: "attached observer requires driver context".to_owned(),
            })
        }

        fn observe_with_context(
            &self,
            attempt: &DecodedAttempt<Exposition, ()>,
            context: &NativeAttemptContext,
        ) -> Result<(), NativeObserverError> {
            assert_eq!(attempt.facts.outcome, SourceOutcome::Success);
            self.events.borrow_mut().push("native");
            self.contexts.borrow_mut().push(context.clone());
            Ok(())
        }
    }

    #[derive(Debug)]
    struct RejectingAttachedOwner {
        events: Rc<RefCell<Vec<&'static str>>>,
        attempts: RefCell<Vec<ArchiveAttemptObservation>>,
        losses: RefCell<Vec<ArchiveIssuedLossObservation>>,
    }

    impl AttachedArchiveAttemptOwner for RejectingAttachedOwner {
        fn try_observe_attempt(
            &self,
            observation: ArchiveAttemptObservation,
        ) -> Result<AttachedAttemptAdmission, AdmissionRejection> {
            self.events.borrow_mut().push("archive_admission");
            self.attempts.borrow_mut().push(observation);
            Err(AdmissionRejection::Capacity)
        }

        fn record_visible_loss(
            &self,
            observation: ArchiveIssuedLossObservation,
        ) -> Result<(), ArchiveOwnerError> {
            self.events.borrow_mut().push("visible_loss");
            self.losses.borrow_mut().push(observation);
            Ok(())
        }

        fn record_missed(
            &self,
            _observation: ArchiveMissedObservation,
        ) -> Result<(), ArchiveOwnerError> {
            Ok(())
        }
    }

    #[derive(Debug, Default)]
    struct RecordingOwner {
        events: Rc<RefCell<Vec<&'static str>>>,
        attempts: RefCell<Vec<ArchiveAttemptObservation>>,
        missed: RefCell<Vec<ArchiveMissedObservation>>,
    }

    #[async_trait(?Send)]
    impl ArchiveAttemptOwner for RecordingOwner {
        async fn observe_attempt(
            &self,
            observation: ArchiveAttemptObservation,
        ) -> Result<(), ArchiveOwnerError> {
            self.events.borrow_mut().push("archive");
            self.attempts.borrow_mut().push(observation);
            Ok(())
        }

        async fn observe_missed(
            &self,
            observation: ArchiveMissedObservation,
        ) -> Result<(), ArchiveOwnerError> {
            self.missed.borrow_mut().push(observation);
            Ok(())
        }
    }

    fn fetched(body: &'static [u8]) -> FetchedAttempt {
        FetchedAttempt {
            source_id: "source-a".to_owned(),
            source_record_seq: 0,
            request_attempt_seq: Some(0),
            scheduled_ns: Some(0),
            request_start_ns: Some(0),
            first_byte_ns: Some(1),
            capture_ns: Some(1),
            latency_ns: Some(1),
            disposition: FetchDisposition::Response {
                status: 200,
                content_type: Some("text/plain; version=0.0.4".to_owned()),
                content_encoding: None,
                encoded_body: Bytes::from_static(body),
                decoded_body: Bytes::from_static(body),
            },
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn native_delivery_precedes_archive_handoff_and_releases_capacity() {
        let events = Rc::new(RefCell::new(Vec::new()));
        let owner = Rc::new(RecordingOwner {
            events: events.clone(),
            ..RecordingOwner::default()
        });
        let native = Rc::new(OrderingNative {
            events: events.clone(),
        });
        let clock: Rc<dyn Clock> = Rc::new(SimClock::new());
        let limits = DecodeLimits::default();
        let pool = BoundedTelemetryDecodePool::new(1).unwrap();
        let decoder: Arc<dyn AttemptDecoder<Exposition, ()>> =
            Arc::new(PrometheusAttemptDecoder::new(
                Arc::new(StrictExpositionParser),
                Arc::new(NoopNativeEntityDecoder),
            ));
        let pipeline = PrometheusAttemptPipeline::new(
            clock,
            decoder,
            limits,
            pool.clone(),
            native,
            owner.clone(),
        )
        .unwrap();

        pipeline
            .observe_attempt(fetched(b"# TYPE temperature gauge\ntemperature 3\n"))
            .await
            .unwrap();

        assert_eq!(&*events.borrow(), &["native", "archive"]);
        assert_eq!(pool.available_permits(), 1);
        assert_eq!(owner.attempts.borrow().len(), 1);
        assert_eq!(owner.attempts.borrow()[0].parse_done_ns, 0);
        assert_eq!(owner.attempts.borrow()[0].archive_enqueue_ns, 0);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn parse_failure_is_still_one_all_outcome_archive_observation() {
        let owner = Rc::new(RecordingOwner::default());
        let clock: Rc<dyn Clock> = Rc::new(SimClock::new());
        let pipeline = PrometheusAttemptPipeline::strict_standalone(
            clock,
            Arc::new(StrictExpositionParser),
            DecodeLimits::default(),
            BoundedTelemetryDecodePool::new(1).unwrap(),
            owner.clone(),
        )
        .unwrap();

        pipeline
            .observe_attempt(fetched(b"broken{ 1\n"))
            .await
            .unwrap();

        assert_eq!(owner.attempts.borrow().len(), 1);
        assert_eq!(
            owner.attempts.borrow()[0].decoded.facts.outcome,
            SourceOutcome::Parse
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn attached_boundary_decodes_once_delivers_native_context_then_records_rejection() {
        let events = Rc::new(RefCell::new(Vec::new()));
        let calls = Arc::new(AtomicUsize::new(0));
        let decoder: Arc<dyn AttemptDecoder<Exposition, ()>> = Arc::new(CountingDecoder {
            calls: calls.clone(),
            inner: PrometheusAttemptDecoder::new(
                Arc::new(StrictExpositionParser),
                Arc::new(NoopNativeEntityDecoder),
            ),
        });
        let native = Rc::new(ContextNative {
            events: events.clone(),
            contexts: RefCell::new(Vec::new()),
        });
        let owner = Rc::new(RejectingAttachedOwner {
            events: events.clone(),
            attempts: RefCell::new(Vec::new()),
            losses: RefCell::new(Vec::new()),
        });
        let boundary = BoundaryReference {
            transition_id: "warmup-to-profile".to_owned(),
            boundary_id: "source-a-end".to_owned(),
            phase_id: "warmup".to_owned(),
            source_id: "source-a".to_owned(),
            role: aiperf_telemetry_archive::BoundaryRole::PhaseEnd,
            coalescing_group_id: None,
        };
        let pipeline = PrometheusAttemptPipeline::new_attached(
            Rc::new(SimClock::new()),
            decoder,
            DecodeLimits::default(),
            BoundedTelemetryDecodePool::new(1).unwrap(),
            native.clone(),
            owner.clone(),
        )
        .unwrap();

        let disposition = pipeline
            .observe_attempt_envelope(TelemetryAttemptEnvelope {
                attempt: fetched(b"# TYPE temperature gauge\ntemperature 3\n"),
                reason: ScrapeReasonV1::Boundary,
                boundary_refs: vec![boundary.clone()],
                active_phase_ids: BTreeSet::from(["profiling".to_owned(), "warmup".to_owned()]),
            })
            .await
            .unwrap();

        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(
            &*events.borrow(),
            &["native", "archive_admission", "visible_loss"]
        );
        assert_eq!(
            disposition,
            TelemetryAttemptDisposition::Loss {
                kind: LossKindV1::ArchiveRejected,
                reason: LossReasonV1::ArchiveAdmissionRejected,
            }
        );
        assert_eq!(native.contexts.borrow().len(), 1);
        assert_eq!(
            native.contexts.borrow()[0].active_phase_ids,
            BTreeSet::from(["profiling".to_owned(), "warmup".to_owned()])
        );
        assert_eq!(
            native.contexts.borrow()[0].boundary_refs,
            vec![boundary.clone()]
        );
        assert_eq!(owner.attempts.borrow().len(), 1);
        assert_eq!(
            owner.attempts.borrow()[0].projection_context.boundary_refs,
            vec![boundary.clone()]
        );
        assert_eq!(owner.losses.borrow().len(), 1);
        assert_eq!(owner.losses.borrow()[0].boundary_refs, vec![boundary]);
    }
}
