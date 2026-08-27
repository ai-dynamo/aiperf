// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Boundary-owned capture projections an exporter may require.
//!
//! An exporter plugin declares what it needs from a finished run — the
//! finalized report, the exact per-record projection, or a folded histogram
//! projection — and the host supplies exactly that. The projections are defined
//! here, once, so the requirement an exporter declares and the value it receives
//! are the same Rust type. `aiperf-plugin-api` references these types; it never
//! redefines them.
//!
//! Every projection is schema-tagged and version-tagged. A projection whose
//! schema or version does not match this crate's constants is refused rather
//! than reinterpreted: a silently-reinterpreted capture is a wrong artifact, not
//! a degraded one.
//!
//! Ordering is part of each projection's identity. [`ExactRecordsV1`] is ordered
//! by dispatch start and then record index; [`GenAiClientHistogramsV1`] is
//! ordered by metric emission order and then by dimension. Two hosts that
//! observe the same run therefore serialize byte-identical captures.

use std::collections::BTreeMap;
use std::fmt::{self, Display, Formatter};

use serde::{Deserialize, Serialize};

use crate::histogram::{GenAiHistogramMetric, TokenDirection, bucket_index, is_observable};

/// Schema identifier shared by every capture projection in this module.
pub const CAPTURE_PROJECTION_SCHEMA: &str = "aiperf.capture";

/// Version of [`CAPTURE_PROJECTION_SCHEMA`] this crate defines.
pub const CAPTURE_PROJECTION_VERSION: u32 = 1;

/// Why a capture projection was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CaptureError {
    /// The projection's schema tag is not [`CAPTURE_PROJECTION_SCHEMA`].
    UnknownSchema {
        /// The schema tag carried by the refused projection.
        found: String,
    },
    /// The projection's version is not [`CAPTURE_PROJECTION_VERSION`].
    UnsupportedVersion {
        /// The version carried by the refused projection.
        found: u32,
    },
    /// A histogram's bucket-count vector is not `bounds.len() + 1` long.
    BucketCountMismatch {
        /// Number of explicit boundaries.
        bounds: usize,
        /// Number of bucket counts supplied.
        buckets: usize,
    },
    /// Two histograms with different boundaries were merged.
    BoundsMismatch,
    /// The records or series are not in their projection's canonical order.
    OrderingViolation,
}

impl Display for CaptureError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownSchema { found } => {
                write!(formatter, "unknown capture schema {found:?}")
            }
            Self::UnsupportedVersion { found } => {
                write!(formatter, "unsupported capture version {found}")
            }
            Self::BucketCountMismatch { bounds, buckets } => write!(
                formatter,
                "histogram has {bounds} bounds but {buckets} bucket counts"
            ),
            Self::BoundsMismatch => {
                formatter.write_str("cannot merge histograms with different boundaries")
            }
            Self::OrderingViolation => {
                formatter.write_str("capture projection is not in canonical order")
            }
        }
    }
}

impl std::error::Error for CaptureError {}

/// One metric value exactly as the run projected it for a record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetricValueV1 {
    /// The projected value in its display unit.
    pub value: f64,
    /// The display unit, as [`crate::histogram::seconds_scale`] reads it.
    pub unit: String,
}

/// One request's exact per-record projection.
///
/// The map is a `BTreeMap` so a serialized record's key order is the metric
/// name order rather than an insertion order that depends on which transport
/// produced it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExactRecordV1 {
    /// Monotonic index assigned when the record was captured.
    pub record_index: u64,
    /// Conversation this record belongs to, when the workload has one.
    pub conversation_id: Option<String>,
    /// Zero-based turn within the conversation.
    pub turn_index: u32,
    /// Model named on the request, when the endpoint carries one.
    pub model: Option<String>,
    /// Clock-ns when dispatch started.
    pub start_ns: i64,
    /// Clock-ns when the request reached terminal, if it did.
    pub end_ns: Option<i64>,
    /// The record's `error.type` classification; `None` on success.
    pub error_type: Option<String>,
    /// Projected metric values keyed by report metric name.
    pub metrics: BTreeMap<String, MetricValueV1>,
}

impl ExactRecordV1 {
    /// The canonical ordering key: dispatch start, then capture index.
    pub fn order_key(&self) -> (i64, u64) {
        (self.start_ns, self.record_index)
    }

    /// Whether the record completed without a classified failure.
    pub fn is_success(&self) -> bool {
        self.error_type.is_none()
    }
}

/// The exact per-record capture projection for one run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExactRecordsV1 {
    /// Always [`CAPTURE_PROJECTION_SCHEMA`].
    pub schema: String,
    /// Always [`CAPTURE_PROJECTION_VERSION`].
    pub version: u32,
    /// Records in canonical order.
    pub records: Vec<ExactRecordV1>,
}

impl ExactRecordsV1 {
    /// Build a projection from records in arbitrary order, sorting canonically.
    pub fn from_records(mut records: Vec<ExactRecordV1>) -> Self {
        records.sort_by_key(ExactRecordV1::order_key);
        Self {
            schema: CAPTURE_PROJECTION_SCHEMA.to_owned(),
            version: CAPTURE_PROJECTION_VERSION,
            records,
        }
    }

    /// Refuse a projection whose schema, version, or ordering is not this one.
    pub fn validate(&self) -> Result<(), CaptureError> {
        validate_tags(&self.schema, self.version)?;
        if self.records.windows(2).any(|pair| {
            let [earlier, later] = pair else {
                return false;
            };
            earlier.order_key() > later.order_key()
        }) {
            return Err(CaptureError::OrderingViolation);
        }
        Ok(())
    }
}

/// One explicit-bucket histogram.
///
/// `bucket_counts` is always `bounds.len() + 1` long — the OTLP invariant — with
/// the trailing entry counting observations above every boundary. `min` and
/// `max` stay absent until the first observation so an empty histogram never
/// claims a bound it never saw.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExplicitHistogramV1 {
    /// Explicit upper boundaries in ascending order.
    pub bounds: Vec<f64>,
    /// Per-bucket counts, `bounds.len() + 1` long.
    pub bucket_counts: Vec<u64>,
    /// Total admitted observations.
    pub count: u64,
    /// Sum of admitted observations.
    pub sum: f64,
    /// Smallest admitted observation.
    pub min: Option<f64>,
    /// Largest admitted observation.
    pub max: Option<f64>,
}

impl ExplicitHistogramV1 {
    /// An empty histogram over `bounds`.
    pub fn new(bounds: &[f64]) -> Self {
        Self {
            bounds: bounds.to_vec(),
            bucket_counts: vec![0; bounds.len() + 1],
            count: 0,
            sum: 0.0,
            min: None,
            max: None,
        }
    }

    /// Admit one observation under the shared first-upper-bound rule.
    ///
    /// Non-finite values are dropped; see
    /// [`crate::histogram::is_observable`].
    pub fn observe(&mut self, value: f64) {
        if !is_observable(value) {
            return;
        }
        self.bucket_counts[bucket_index(&self.bounds, value)] += 1;
        self.count += 1;
        self.sum += value;
        self.min = Some(self.min.map_or(value, |current| current.min(value)));
        self.max = Some(self.max.map_or(value, |current| current.max(value)));
    }

    /// Merge a same-bounds histogram into this one.
    pub fn merge(&mut self, other: &Self) -> Result<(), CaptureError> {
        if self.bounds != other.bounds {
            return Err(CaptureError::BoundsMismatch);
        }
        for (accumulated, incoming) in self.bucket_counts.iter_mut().zip(&other.bucket_counts) {
            *accumulated += incoming;
        }
        self.count += other.count;
        self.sum += other.sum;
        if let Some(value) = other.min {
            self.min = Some(self.min.map_or(value, |current| current.min(value)));
        }
        if let Some(value) = other.max {
            self.max = Some(self.max.map_or(value, |current| current.max(value)));
        }
        Ok(())
    }

    /// Refuse a histogram that violates the OTLP bucket-length invariant.
    pub fn validate(&self) -> Result<(), CaptureError> {
        if self.bucket_counts.len() != self.bounds.len() + 1 {
            return Err(CaptureError::BucketCountMismatch {
                bounds: self.bounds.len(),
                buckets: self.bucket_counts.len(),
            });
        }
        Ok(())
    }
}

/// The attribute dimension one histogram series is emitted under.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct HistogramDimensionV1 {
    /// The `gen_ai.token.type` value, on token-usage series only.
    pub token_type: Option<String>,
    /// The `error.type` value, on duration series only; `None` on success.
    pub error_type: Option<String>,
}

impl HistogramDimensionV1 {
    /// The success dimension of a duration series.
    pub fn success() -> Self {
        Self {
            token_type: None,
            error_type: None,
        }
    }

    /// The dimension of one token-usage direction.
    pub fn token(direction: TokenDirection) -> Self {
        Self {
            token_type: Some(direction.attribute_value().to_owned()),
            error_type: None,
        }
    }

    /// The dimension of a duration series discriminated on `error.type`.
    pub fn error(error_type: impl Into<String>) -> Self {
        Self {
            token_type: None,
            error_type: Some(error_type.into()),
        }
    }
}

/// One metric stream at one dimension.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HistogramSeriesV1 {
    /// The semantic-convention metric name, from
    /// [`GenAiHistogramMetric::spec_name`].
    pub metric: String,
    /// The semantic-convention unit, from [`GenAiHistogramMetric::unit`].
    pub unit: String,
    /// The attribute dimension the series is emitted under.
    pub dimension: HistogramDimensionV1,
    /// The accumulated histogram.
    pub histogram: ExplicitHistogramV1,
}

impl HistogramSeriesV1 {
    /// An empty series for `metric` at `dimension`, using the metric's bounds.
    pub fn new(metric: GenAiHistogramMetric, dimension: HistogramDimensionV1) -> Self {
        Self {
            metric: metric.spec_name().to_owned(),
            unit: metric.unit().to_owned(),
            dimension,
            histogram: ExplicitHistogramV1::new(metric.bounds()),
        }
    }

    /// The canonical ordering key: metric emission order, then dimension.
    pub fn order_key(&self) -> (usize, &HistogramDimensionV1) {
        let position = crate::histogram::GEN_AI_HISTOGRAM_METRICS
            .iter()
            .position(|metric| metric.spec_name() == self.metric)
            .unwrap_or(usize::MAX);
        (position, &self.dimension)
    }
}

/// The folded GenAI client histogram projection for one run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenAiClientHistogramsV1 {
    /// Always [`CAPTURE_PROJECTION_SCHEMA`].
    pub schema: String,
    /// Always [`CAPTURE_PROJECTION_VERSION`].
    pub version: u32,
    /// Series in canonical order.
    pub series: Vec<HistogramSeriesV1>,
}

impl GenAiClientHistogramsV1 {
    /// Build a projection from series in arbitrary order, sorting canonically.
    pub fn from_series(mut series: Vec<HistogramSeriesV1>) -> Self {
        series.sort_by(|left, right| left.order_key().cmp(&right.order_key()));
        Self {
            schema: CAPTURE_PROJECTION_SCHEMA.to_owned(),
            version: CAPTURE_PROJECTION_VERSION,
            series,
        }
    }

    /// Refuse a projection whose schema, version, ordering, or bucket lengths
    /// are not this one's.
    pub fn validate(&self) -> Result<(), CaptureError> {
        validate_tags(&self.schema, self.version)?;
        for series in &self.series {
            series.histogram.validate()?;
        }
        if self.series.windows(2).any(|pair| {
            let [earlier, later] = pair else {
                return false;
            };
            earlier.order_key() > later.order_key()
        }) {
            return Err(CaptureError::OrderingViolation);
        }
        Ok(())
    }
}

/// The finalized report projection every exporter receives.
///
/// The value is whatever projection the run finalized, carried opaquely: an
/// exporter reads it, and the host commits the identical bytes through
/// [`crate::report::write_finalized_report_json`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FinalReportV1 {
    /// Always [`CAPTURE_PROJECTION_SCHEMA`].
    pub schema: String,
    /// Always [`CAPTURE_PROJECTION_VERSION`].
    pub version: u32,
    /// The finalized report projection.
    pub report: serde_json::Value,
}

impl FinalReportV1 {
    /// Tag one finalized report projection.
    pub fn new(report: serde_json::Value) -> Self {
        Self {
            schema: CAPTURE_PROJECTION_SCHEMA.to_owned(),
            version: CAPTURE_PROJECTION_VERSION,
            report,
        }
    }

    /// Refuse a projection whose schema or version is not this one.
    pub fn validate(&self) -> Result<(), CaptureError> {
        validate_tags(&self.schema, self.version)
    }
}

fn validate_tags(schema: &str, version: u32) -> Result<(), CaptureError> {
    if schema != CAPTURE_PROJECTION_SCHEMA {
        return Err(CaptureError::UnknownSchema {
            found: schema.to_owned(),
        });
    }
    if version != CAPTURE_PROJECTION_VERSION {
        return Err(CaptureError::UnsupportedVersion { found: version });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(index: u64, start_ns: i64) -> ExactRecordV1 {
        ExactRecordV1 {
            record_index: index,
            conversation_id: None,
            turn_index: 0,
            model: None,
            start_ns,
            end_ns: Some(start_ns + 1),
            error_type: None,
            metrics: BTreeMap::new(),
        }
    }

    #[test]
    fn records_sort_by_dispatch_start_then_index() {
        let projection = ExactRecordsV1::from_records(vec![record(2, 30), record(0, 10)]);
        assert_eq!(projection.records[0].record_index, 0);
        assert!(projection.validate().is_ok());
    }

    #[test]
    fn a_mismatched_schema_is_refused_rather_than_reinterpreted() {
        let mut projection = ExactRecordsV1::from_records(vec![record(0, 0)]);
        projection.schema = "other".to_owned();
        assert!(matches!(
            projection.validate(),
            Err(CaptureError::UnknownSchema { .. })
        ));
    }

    #[test]
    fn histograms_merge_only_across_identical_bounds() {
        let mut left = ExplicitHistogramV1::new(&[1.0, 2.0]);
        left.observe(1.0);
        left.observe(f64::NAN);
        assert_eq!(left.count, 1);
        assert_eq!(left.bucket_counts, vec![1, 0, 0]);

        let right = ExplicitHistogramV1::new(&[1.0, 2.0]);
        assert!(left.merge(&right).is_ok());
        assert!(matches!(
            left.merge(&ExplicitHistogramV1::new(&[1.0])),
            Err(CaptureError::BoundsMismatch)
        ));
    }
}
