// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Parser resource limits enforced before archive projection.

/// Hard bounds for one decoded exposition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseLimits {
    /// Maximum decoded body bytes.
    pub max_decoded_bytes: usize,
    /// Maximum physical lines, including comments and metadata.
    pub max_lines: usize,
    /// Maximum bytes in one physical line, excluding the line feed.
    pub max_line_bytes: usize,
    /// Maximum metric families, including metadata-only families.
    pub max_families: usize,
    /// Maximum distinct metrics across all families.
    pub max_metrics: usize,
    /// Maximum structured metric points across all metrics.
    pub max_metric_points: usize,
    /// Maximum emitted wire samples across the exposition.
    pub max_wire_samples: usize,
    /// Maximum labels on one sample before role-label extraction.
    pub max_labels_per_sample: usize,
    /// Maximum UTF-8 bytes in one label name.
    pub max_label_name_bytes: usize,
    /// Maximum UTF-8 bytes in one decoded label value.
    pub max_label_value_bytes: usize,
    /// Maximum UTF-8 bytes in one decoded HELP or UNIT value.
    pub max_metadata_value_bytes: usize,
    /// Maximum bytes in one numeric or timestamp lexeme.
    pub max_numeric_lexeme_bytes: usize,
    /// Maximum histogram buckets in one metric point.
    pub max_buckets_per_point: usize,
    /// Maximum summary quantiles in one metric point.
    pub max_quantiles_per_point: usize,
    /// Maximum StateSet states in one metric point.
    pub max_states_per_point: usize,
    /// Maximum exemplars in one exposition.
    pub max_exemplars: usize,
    /// Maximum labels in one exemplar.
    pub max_exemplar_labels: usize,
    /// Maximum combined Unicode scalar count across exemplar label names and values.
    pub max_exemplar_label_codepoints: usize,
}

impl Default for ParseLimits {
    fn default() -> Self {
        Self {
            max_decoded_bytes: 32 * 1024 * 1024,
            max_lines: 1_000_000,
            max_line_bytes: 1024 * 1024,
            max_families: 100_000,
            max_metrics: 1_000_000,
            max_metric_points: 1_000_000,
            max_wire_samples: 2_000_000,
            max_labels_per_sample: 128,
            max_label_name_bytes: 1024,
            max_label_value_bytes: 64 * 1024,
            max_metadata_value_bytes: 1024 * 1024,
            max_numeric_lexeme_bytes: 4096,
            max_buckets_per_point: 100_000,
            max_quantiles_per_point: 100_000,
            max_states_per_point: 100_000,
            max_exemplars: 1_000_000,
            max_exemplar_labels: 128,
            max_exemplar_label_codepoints: 128,
        }
    }
}
