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
    serde_yaml::from_str(include_str!("../../resources/public_datasets.yaml"))
        .expect("embedded public_datasets.yaml is valid")
});

/// Look up a public dataset by name.
pub fn lookup(name: &str) -> Option<&'static PublicMeta> {
    CATALOG.get(name)
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

    /// Guard the stringly-typed YAML catalog: every entry must name a format that
    /// the runtime can resolve — either a registered linear loader or a known
    /// Graph-IR input format — and carry that format's required options, so a
    /// typo'd format or a missing `prompt_column`/`conversation_column` fails here
    /// instead of at runtime when a user selects the dataset.
    #[test]
    fn every_entry_has_a_resolvable_format_and_required_options() {
        use aiperf_runtime::dataset::loader::LoaderRegistry;

        // Formats resolved through the engine's graph-input path
        // (`engine/graph_input.rs`) rather than the linear loader registry.
        const GRAPH_INPUT_FORMATS: &[&str] = &[
            "weka_trace",
            "dag_jsonl",
            "dynamo_trace",
            "conditional_graph",
        ];

        let registry = LoaderRegistry::with_builtin_formats().expect("builtin formats register");
        for (name, meta) in CATALOG.iter() {
            let resolvable = registry.get(&meta.format).is_ok()
                || GRAPH_INPUT_FORMATS.contains(&meta.format.as_str());
            assert!(
                resolvable,
                "catalog entry {name:?} uses unresolvable format {:?} \
                 (not a registered loader nor a known graph-input format)",
                meta.format
            );
            match meta.format.as_str() {
                "hf_instruction_response" => assert!(
                    meta.options.contains_key("prompt_column")
                        || meta.options.contains_key("prompt_template"),
                    "catalog entry {name:?} (hf_instruction_response) needs a \
                     `prompt_column` or `prompt_template` option"
                ),
                "hf_conversation" => assert!(
                    meta.options.contains_key("conversation_column"),
                    "catalog entry {name:?} (hf_conversation) needs a `conversation_column` option"
                ),
                _ => {}
            }
        }
    }
}
