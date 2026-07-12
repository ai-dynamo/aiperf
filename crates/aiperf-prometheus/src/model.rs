// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Lossless exposition and structured MetricPoint model.

use std::collections::BTreeMap;

use crate::format::ExpositionFormat;
use crate::number::{CreatedTimestamp, ExactNumber, SourceTimestamp};

/// Canonically ordered decoded label map.
pub type LabelSet = BTreeMap<String, String>;

/// Semantic metric-family type after format-specific role resolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SemanticType {
    /// OpenMetrics unknown or Prometheus untyped family.
    Unknown,
    /// Point-in-time scalar gauge.
    Gauge,
    /// Monotonic total with optional creation time and exemplar.
    Counter,
    /// Related named boolean states.
    StateSet,
    /// Textual information encoded as merged labels and value one.
    Info,
    /// Cumulative event distribution.
    Histogram,
    /// Current distribution with gauge-shaped sum/count semantics.
    GaugeHistogram,
    /// Count/sum and optional precomputed quantiles.
    Summary,
}

/// One exact metadata directive.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetadataLine {
    /// Decoded directive value, which may be empty.
    pub value: String,
    /// One-based source line.
    pub line: usize,
}

/// Complete, atomically parsed exposition.
#[derive(Debug, Clone, PartialEq)]
pub struct Exposition {
    /// Grammar used for this document.
    pub format: ExpositionFormat,
    /// Families in first source occurrence order, including metadata-only families.
    pub families: Vec<MetricFamily>,
    /// Number of emitted wire samples retained in all points.
    pub wire_sample_count: usize,
}

impl Exposition {
    /// Returns one family by its exact source family name.
    pub fn family(&self, name: &str) -> Option<&MetricFamily> {
        self.families.iter().find(|family| family.name == name)
    }

    /// Returns the total structured metric-point count.
    pub fn metric_point_count(&self) -> usize {
        self.families
            .iter()
            .flat_map(|family| &family.metrics)
            .map(|metric| metric.points.len())
            .sum()
    }
}

/// One parsed family with exact metadata and structured metrics.
#[derive(Debug, Clone, PartialEq)]
pub struct MetricFamily {
    /// Exact source family name selected by the format role matrix.
    pub name: String,
    /// Exact emitted TYPE token, or the format's synthesized unknown token when absent.
    pub source_type_token: String,
    /// Resolved semantic type.
    pub semantic_type: SemanticType,
    /// Exact HELP directive when emitted.
    pub help: Option<MetadataLine>,
    /// One-based TYPE line when emitted.
    pub type_line: Option<usize>,
    /// Exact UNIT directive when emitted.
    pub unit: Option<MetadataLine>,
    /// Metrics in first source occurrence order.
    pub metrics: Vec<Metric>,
    /// Source order of this family.
    pub family_seq: u64,
}

/// One unique metric identity with one or more ordered points.
#[derive(Debug, Clone, PartialEq)]
pub struct Metric {
    /// Complete format-level identity labels after component-label extraction.
    pub labels: LabelSet,
    /// Ordered points for this identity.
    pub points: Vec<MetricPoint>,
}

/// One ordered semantic MetricPoint and all wire evidence that produced it.
#[derive(Debug, Clone, PartialEq)]
pub struct MetricPoint {
    /// Source order by the first contributing wire sample.
    pub metric_point_seq: u64,
    /// Identity labels copied from the containing metric for convenient projection.
    pub labels: LabelSet,
    /// Relationship among contributing component timestamps.
    pub point_time_status: PointTimeStatus,
    /// Common first-wire timestamp only when all components explicitly agree.
    pub source_timestamp: SourceTimestamp,
    /// Structured semantic value.
    pub value: MetricValue,
    /// Every contributing wire sample in exact source order.
    pub wire_samples: Vec<WireSample>,
}

/// Component-timestamp relationship for one structured point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PointTimeStatus {
    /// No component emitted a timestamp.
    AllAbsent,
    /// Every component emitted the same exact timestamp.
    UniformExplicit,
    /// Every component emitted a timestamp, but values differ.
    MixedComponents,
    /// Some components emitted timestamps and others did not.
    PartialComponents,
}

/// Source role of one retained sample line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WireSampleRole {
    /// Unknown or gauge scalar.
    Scalar,
    /// Counter total.
    CounterTotal,
    /// Counter semantic creation timestamp.
    CounterCreated,
    /// StateSet state value.
    State,
    /// Info merged-label value.
    Info,
    /// Histogram cumulative bucket.
    HistogramBucket,
    /// Histogram event sum.
    HistogramSum,
    /// Histogram emitted count.
    HistogramCount,
    /// Histogram semantic creation timestamp.
    HistogramCreated,
    /// Gauge-histogram cumulative bucket.
    GaugeHistogramBucket,
    /// Gauge-histogram sum.
    GaugeHistogramSum,
    /// Gauge-histogram emitted count.
    GaugeHistogramCount,
    /// Summary count.
    SummaryCount,
    /// Summary sum.
    SummarySum,
    /// Summary semantic creation timestamp.
    SummaryCreated,
    /// Summary quantile.
    SummaryQuantile,
}

/// One exact emitted sample and optional exemplar.
#[derive(Debug, Clone, PartialEq)]
pub struct WireSample {
    /// One-based source line.
    pub line: usize,
    /// Exact emitted sample name.
    pub emitted_name: String,
    /// Format-resolved semantic role.
    pub role: WireSampleRole,
    /// Complete decoded labels as emitted, in canonical key order.
    pub labels: LabelSet,
    /// Exact source value.
    pub value: ExactNumber,
    /// Exact optional sample timestamp.
    pub source_timestamp: SourceTimestamp,
    /// Optional sample-owned exemplar.
    pub exemplar: Option<Exemplar>,
}

/// One exact scalar or bucket exemplar.
#[derive(Debug, Clone, PartialEq)]
pub struct Exemplar {
    /// Canonically ordered decoded exemplar labels.
    pub labels: LabelSet,
    /// Exact exemplar value.
    pub value: ExactNumber,
    /// Exact optional exemplar timestamp.
    pub timestamp: SourceTimestamp,
}

/// Structured semantic payload selected by family type.
#[derive(Debug, Clone, PartialEq)]
pub enum MetricValue {
    /// Unknown or gauge scalar.
    Scalar {
        /// Exact scalar value.
        value: ExactNumber,
        /// Scalar-owned exemplar when the role permits one.
        exemplar: Option<Exemplar>,
    },
    /// Counter total and creation facts.
    Counter(CounterValue),
    /// Ordered related states.
    StateSet(Vec<StateValue>),
    /// Merged Info labels without an invented abstract partition.
    Info(InfoValue),
    /// Histogram or gauge-histogram payload.
    Histogram(HistogramValue),
    /// Summary payload.
    Summary(SummaryValue),
}

/// Structured counter payload.
#[derive(Debug, Clone, PartialEq)]
pub struct CounterValue {
    /// Exact counter total.
    pub total: ExactNumber,
    /// Optional semantic creation timestamp.
    pub created: CreatedTimestamp,
    /// Optional total-owned exemplar.
    pub exemplar: Option<Exemplar>,
}

/// One StateSet state in source order.
#[derive(Debug, Clone, PartialEq)]
pub struct StateValue {
    /// Decoded state name from the family-named role label.
    pub state: String,
    /// Exact zero-or-one source value.
    pub enabled: ExactNumber,
}

/// Whether text Info labels have an external analytical partition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InfoLabelPartitionStatus {
    /// The text wire format cannot distinguish metric labels from Info value labels.
    UnavailableFromText,
    /// A named persisted policy supplied a disjoint analytical partition.
    PolicyApplied,
}

/// Lossless text-native Info payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InfoValue {
    /// Complete merged label set used as source identity.
    pub wire_merged_labels: LabelSet,
    /// Optional policy-produced metric-label partition.
    pub partitioned_metric_labels: Option<LabelSet>,
    /// Optional policy-produced value-label partition.
    pub partitioned_value_labels: Option<LabelSet>,
    /// Optional persisted partition policy ID.
    pub partition_policy_id: Option<String>,
    /// Availability of the abstract partition.
    pub partition_status: InfoLabelPartitionStatus,
}

/// Origin of a structured histogram count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CountOrigin {
    /// An emitted count role matched the positive-infinity bucket.
    EmittedAndValidated,
    /// OpenMetrics omitted count and the positive-infinity bucket supplied it.
    DerivedFromPositiveInfinity,
}

/// Structured histogram or gauge-histogram payload.
#[derive(Debug, Clone, PartialEq)]
pub struct HistogramValue {
    /// Optional exact event/current-distribution sum.
    pub sum: ExactNumber,
    /// Exact total count.
    pub count: ExactNumber,
    /// Whether count was emitted or derived.
    pub count_origin: CountOrigin,
    /// Optional semantic creation timestamp.
    pub created: CreatedTimestamp,
    /// Numerically ordered cumulative buckets.
    pub buckets: Vec<HistogramBucket>,
}

/// One cumulative histogram bucket.
#[derive(Debug, Clone, PartialEq)]
pub struct HistogramBucket {
    /// Exact decoded `le` label lexeme.
    pub upper_bound_lexeme: String,
    /// Exact numeric upper bound.
    pub upper_bound: ExactNumber,
    /// Exact cumulative count.
    pub cumulative_count: ExactNumber,
    /// Optional bucket-owned exemplar.
    pub exemplar: Option<Exemplar>,
}

/// Structured summary payload.
#[derive(Debug, Clone, PartialEq)]
pub struct SummaryValue {
    /// Optional exact event sum.
    pub sum: ExactNumber,
    /// Optional exact event count.
    pub count: ExactNumber,
    /// Optional semantic creation timestamp.
    pub created: CreatedTimestamp,
    /// Numerically ordered quantiles.
    pub quantiles: Vec<QuantileValue>,
}

/// One summary quantile/value pair.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantileValue {
    /// Exact decoded `quantile` label lexeme.
    pub quantile_lexeme: String,
    /// Exact quantile in the inclusive interval zero through one.
    pub quantile: ExactNumber,
    /// Exact quantile value.
    pub value: ExactNumber,
}
