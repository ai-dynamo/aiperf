// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! The public-dataset catalog — per-dataset source/format/option metadata.
//!
//! Ported from `src/aiperf/orchestrator/rust_wire.py::_public_dataset`, whose
//! per-dataset metadata comes from the Python plugin registry
//! (`get_public_dataset_loader_metadata` + `_PUBLIC_NATIVE_FORMATS`). That
//! metadata is static, so it is captured once into
//! `resources/public_datasets.json` and embedded here (like the metric
//! metadata). The runtime-derived `max_conversations` option is computed by the
//! loader, not stored.

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
    serde_json::from_str(include_str!("../../resources/public_datasets.json"))
        .expect("embedded public_datasets.json is valid")
});

/// Look up a public dataset by name.
pub fn lookup(name: &str) -> Option<&'static PublicMeta> {
    CATALOG.get(name)
}

/// Compute `max_conversations` (`_public_max_conversations`): `entries` wins for
/// entries-first loaders; otherwise a streaming loader uses the request cap;
/// otherwise `entries`.
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
