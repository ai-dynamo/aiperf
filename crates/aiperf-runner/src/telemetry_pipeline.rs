// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded fetch-to-native-to-archive attempt pipeline.
//!
//! Decode jobs run in a shared explicitly bounded CPU pool. Their result
//! returns to the source LocalSet, where native delivery occurs synchronously
//! before archive enqueue observation. The sole archive owner remains an
//! injected trait and therefore retains sequencing, projection, and durability
//! authority.

use std::fmt::{self, Debug, Display, Formatter};
use std::sync::Arc;

use aiperf_clock::Clock;
use aiperf_prometheus::Exposition;
use aiperf_telemetry_archive::{
    AttemptDecoder, DecodeLimits, DecodedAttempt, DriverConsumerError, FetchedAttempt,
    MissedCadenceRange, NativeEntityDecoder, NoopNativeEntityDecoder, PrometheusAttemptDecoder,
    TelemetryAttemptConsumer,
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

/// Synchronous native projection/accumulator hook.
pub trait NativeAttemptObserver: Debug {
    /// Deliver a decoded native entity before archive admission or persistence.
    fn observe(&self, attempt: &DecodedAttempt<Exposition, ()>) -> Result<(), NativeObserverError>;
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

/// Prepared source-local consumer sharing global decode capacity and one owner.
pub struct PrometheusAttemptPipeline {
    clock: std::rc::Rc<dyn Clock>,
    decoder: Arc<dyn AttemptDecoder<Exposition, ()>>,
    limits: DecodeLimits,
    decode_pool: BoundedTelemetryDecodePool,
    native: std::rc::Rc<dyn NativeAttemptObserver>,
    owner: std::rc::Rc<dyn ArchiveAttemptOwner>,
}

impl PrometheusAttemptPipeline {
    /// Compose the strict parser, native hook, and archive owner.
    pub fn new(
        clock: std::rc::Rc<dyn Clock>,
        decoder: Arc<dyn AttemptDecoder<Exposition, ()>>,
        limits: DecodeLimits,
        decode_pool: BoundedTelemetryDecodePool,
        native: std::rc::Rc<dyn NativeAttemptObserver>,
        owner: std::rc::Rc<dyn ArchiveAttemptOwner>,
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
            owner,
        })
    }

    /// Compose the stock strict exposition decoder with no native entity.
    pub fn strict_standalone(
        clock: std::rc::Rc<dyn Clock>,
        parser: Arc<dyn aiperf_prometheus::ExpositionParser>,
        limits: DecodeLimits,
        decode_pool: BoundedTelemetryDecodePool,
        owner: std::rc::Rc<dyn ArchiveAttemptOwner>,
    ) -> Result<Self, PipelinePrepareError> {
        let native: Arc<dyn NativeEntityDecoder<()>> = Arc::new(NoopNativeEntityDecoder);
        let decoder: Arc<dyn AttemptDecoder<Exposition, ()>> =
            Arc::new(PrometheusAttemptDecoder::new(parser, native));
        Self::new(
            clock,
            decoder,
            limits,
            decode_pool,
            std::rc::Rc::new(NoopNativeAttemptObserver),
            owner,
        )
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
        let decoded = self
            .decode_pool
            .decode(self.decoder.clone(), attempt, self.limits.clone())
            .await
            .map_err(|error| DriverConsumerError {
                message: error.to_string(),
            })?;
        let parse_done_ns = self.clock.now_ns();
        self.native
            .observe(&decoded)
            .map_err(|error| DriverConsumerError {
                message: error.to_string(),
            })?;
        let archive_enqueue_ns = self.clock.now_ns();
        self.owner
            .observe_attempt(ArchiveAttemptObservation {
                decoded,
                parse_done_ns,
                archive_enqueue_ns,
            })
            .await
            .map_err(|error| DriverConsumerError {
                message: error.to_string(),
            })
    }

    async fn observe_missed(
        &self,
        source_id: &str,
        missed: MissedCadenceRange,
    ) -> Result<(), DriverConsumerError> {
        self.owner
            .observe_missed(ArchiveMissedObservation {
                source_id: source_id.to_owned(),
                missed,
                observed_ns: self.clock.now_ns(),
            })
            .await
            .map_err(|error| DriverConsumerError {
                message: error.to_string(),
            })
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
}
