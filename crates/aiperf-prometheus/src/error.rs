// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Typed, bounded, all-or-nothing parse failures.

use std::fmt::{Display, Formatter, Result as FmtResult};

/// Resource whose configured parser bound was exceeded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LimitKind {
    /// Decoded exposition bytes.
    DecodedBytes,
    /// Physical line count.
    Lines,
    /// One physical line's bytes.
    LineBytes,
    /// Metric-family count.
    Families,
    /// Distinct metric count.
    Metrics,
    /// Structured metric-point count.
    MetricPoints,
    /// Emitted wire-sample count.
    WireSamples,
    /// Labels on one sample.
    LabelsPerSample,
    /// Metric-family or emitted sample name bytes.
    MetricNameBytes,
    /// Label-name bytes.
    LabelNameBytes,
    /// Label-value bytes.
    LabelValueBytes,
    /// HELP or UNIT bytes.
    MetadataValueBytes,
    /// Numeric or timestamp lexeme bytes.
    NumericLexemeBytes,
    /// Buckets in one histogram point.
    BucketsPerPoint,
    /// Quantiles in one summary point.
    QuantilesPerPoint,
    /// States in one StateSet point.
    StatesPerPoint,
    /// Exemplars in one exposition.
    Exemplars,
    /// Labels in one exemplar.
    ExemplarLabels,
    /// Combined exemplar-label Unicode scalar count.
    ExemplarLabelCodepoints,
}

/// Stable failure category for one rejected exposition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseErrorKind {
    /// The decoded entity is not valid UTF-8.
    InvalidUtf8,
    /// A configured resource bound was exceeded.
    LimitExceeded(LimitKind),
    /// Text does not match the selected lexical grammar.
    Syntax,
    /// A metadata directive is malformed, duplicated, or misplaced.
    Metadata,
    /// A numeric sample or timestamp token is invalid for the selected grammar.
    Number,
    /// Labels are malformed or duplicated.
    Label,
    /// An exemplar is malformed or appears on an unsupported role.
    Exemplar,
    /// An OpenMetrics document omitted or misplaced its terminal `# EOF`.
    EndOfFile,
    /// Valid tokens form an invalid metric-family or metric-point role combination.
    Semantic,
    /// The selected advertised format cannot represent a parsed feature.
    UnsupportedFeature,
}

/// One atomic exposition failure with a stable source location.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError {
    /// One-based line, or zero for a body-level error.
    pub line: usize,
    /// One-based byte column, or zero when no column applies.
    pub column: usize,
    /// Stable machine-readable category.
    pub kind: ParseErrorKind,
    /// Bounded human-readable detail.
    pub message: String,
}

impl ParseError {
    /// Constructs one body-level error.
    pub(crate) fn body(kind: ParseErrorKind, message: impl Into<String>) -> Self {
        Self {
            line: 0,
            column: 0,
            kind,
            message: message.into(),
        }
    }

    /// Constructs one line-local error.
    pub(crate) fn line(
        line: usize,
        column: usize,
        kind: ParseErrorKind,
        message: impl Into<String>,
    ) -> Self {
        Self {
            line,
            column,
            kind,
            message: message.into(),
        }
    }
}

impl Display for ParseError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        match (self.line, self.column) {
            (0, _) => formatter.write_str(&self.message),
            (line, 0) => write!(formatter, "line {line}: {}", self.message),
            (line, column) => write!(formatter, "line {line}, column {column}: {}", self.message),
        }
    }
}

impl std::error::Error for ParseError {}
