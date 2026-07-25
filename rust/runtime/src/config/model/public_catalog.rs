// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Embedded public-dataset source, format, and option metadata.
//!
//! `max_conversations` is computed at runtime rather than stored in the catalog.

use std::collections::BTreeMap;
use std::sync::LazyLock;

use serde::Deserialize;

/// Static per-dataset catalog metadata.
#[derive(Clone, Debug, Deserialize)]
pub struct PublicMeta {
    /// Native loader format id.
    pub format: String,
    /// Source coordinates (HuggingFace or URL).
    pub source: serde_json::Value,
    /// Static loader options (columns/multi_turn/template).
    pub options: serde_json::Map<String, serde_json::Value>,
    /// Whether the loader streams rows (affects `max_conversations`).
    pub streaming: bool,
    /// Whether `entries` takes precedence for `max_conversations`.
    pub entries_first: bool,
}

static CATALOG: LazyLock<BTreeMap<String, PublicMeta>> = LazyLock::new(|| {
    serde_yaml::from_str(include_str!("../../../resources/public_datasets.yaml"))
        .expect("embedded public_datasets.yaml is valid")
});

/// Look up a public dataset by name.
pub fn lookup(name: &str) -> Option<&'static PublicMeta> {
    CATALOG.get(name)
}

/// Iterate every catalog entry as `(name, metadata)`.
///
/// Exposed so cross-crate tests (in `aiperf-cli`, which can depend on the
/// runtime loader registry) can validate that every catalog format resolves,
/// without `aiperf-config` itself depending on `aiperf-runtime`.
pub fn catalog_entries() -> impl Iterator<Item = (&'static str, &'static PublicMeta)> {
    CATALOG.iter().map(|(name, meta)| (name.as_str(), meta))
}

/// Compute `max_conversations`.
///
/// Entries-first loaders prefer `entries`; streaming loaders otherwise use the
/// request cap.
pub fn max_conversations(
    meta: &PublicMeta,
    entries: Option<u32>,
    request_count: Option<u64>,
) -> Option<u32> {
    if meta.entries_first && entries.is_some() {
        return entries;
    }
    if meta.streaming && request_count.is_some() {
        return request_count.map(|n| n as u32);
    }
    entries
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sharegpt_is_url_backed() {
        let meta = lookup("sharegpt").expect("sharegpt in catalog");
        assert_eq!(meta.format, "sharegpt");
        assert_eq!(meta.source["type"], serde_json::json!("url"));
    }
}
