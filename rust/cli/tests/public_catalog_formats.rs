// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Cross-crate guard for the embedded public-dataset catalog.
//!
//! The catalog lives in `aiperf_runtime::config`. This test lives in `aiperf-cli`
//! so it can validate every catalog format against the runtime loader registry.
//! Relocated verbatim from the former `aiperf-config` crate's `public_catalog`
//! unit tests when that crate was folded into `aiperf_runtime::config`.

use aiperf_runtime::config::model::public_catalog::catalog_entries;
use aiperf_runtime::dataset::loader::LoaderRegistry;

/// Guard the stringly-typed YAML catalog: every entry must name a format that
/// the runtime can resolve — either a registered linear loader or a known
/// Graph-IR input format — and carry that format's required options, so a
/// typo'd format or a missing `prompt_column`/`conversation_column` fails here
/// instead of at runtime when a user selects the dataset.
#[test]
fn every_entry_has_a_resolvable_format_and_required_options() {
    // Formats resolved through the engine's graph-input path
    // (`engine/graph_input.rs`) rather than the linear loader registry.
    const GRAPH_INPUT_FORMATS: &[&str] = &[
        "weka_trace",
        "dag_jsonl",
        "dynamo_trace",
        "conditional_graph",
    ];

    let registry = LoaderRegistry::with_builtin_formats().expect("builtin formats register");
    for (name, meta) in catalog_entries() {
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
