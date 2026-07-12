// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded, IO-free Prometheus and OpenMetrics exposition parsing.
//!
//! This crate preserves source lexemes, metadata, labels, wire roles,
//! component timestamps, exemplars, and structured metric points. Parsing is
//! atomic: a malformed or semantically invalid exposition returns one typed
//! error and never exposes a partial document. Native server-metrics
//! compatibility remains an explicitly injected projection rather than a
//! parser fallback.

mod compatibility;
mod error;
mod format;
mod limits;
mod model;
mod number;
mod parser;
mod semantic;
mod syntax;

pub use compatibility::{
    CompatibilityProjectionError, NativeCompatibilityProjection, NoopNativeCompatibilityProjection,
};
pub use error::{LimitKind, ParseError, ParseErrorKind};
pub use format::{ContentTypeError, ExpositionFormat};
pub use limits::ParseLimits;
pub use model::{
    CountOrigin, CounterValue, Exemplar, Exposition, HistogramBucket, HistogramValue,
    InfoLabelPartitionStatus, InfoValue, LabelSet, MetadataLine, Metric, MetricFamily, MetricPoint,
    MetricValue, PointTimeStatus, QuantileValue, SemanticType, StateValue, SummaryValue,
    WireSample, WireSampleRole,
};
pub use number::{
    CreatedTimestamp, ExactDecimal, ExactNumber, F64Status, NumberError, NumberKind,
    NumberProduction, SourceTimestamp, TimestampStatus, parse_number_lexeme,
};
pub use parser::{ExpositionParser, StrictExpositionParser};
