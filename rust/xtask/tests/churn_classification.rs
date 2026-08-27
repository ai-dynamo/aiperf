// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pins the type-granular history classification behavior.

use std::collections::BTreeMap;

use aiperf_xtask::abi_churn::{SourceSpan, definition_span, patch_overlaps_type};

#[test]
fn definition_span_ignores_a_doc_comment_name_match() {
    let source = "/// `struct Boundary` is prose.\npub struct Other;\n\npub struct Boundary {\n    field: u64,\n}\n";

    assert_eq!(definition_span(source, "Boundary"), Some((4, 6)));
}

#[test]
fn type_granular_overlap_ignores_co_resident_implementation() {
    let patch = "diff --git a/rust/runtime/src/example.rs b/rust/runtime/src/example.rs\n--- a/rust/runtime/src/example.rs\n+++ b/rust/runtime/src/example.rs\n@@ -70 +70 @@ impl Boundary {\n-old\n+new\n";
    let spans = BTreeMap::from([(
        "rust/runtime/src/example.rs".to_owned(),
        vec![SourceSpan { start: 40, end: 50 }],
    )]);

    assert!(!patch_overlaps_type(patch, &spans, &spans));
}

#[test]
fn type_granular_overlap_detects_an_inserted_field() {
    let patch = "diff --git a/rust/runtime/src/example.rs b/rust/runtime/src/example.rs\n--- a/rust/runtime/src/example.rs\n+++ b/rust/runtime/src/example.rs\n@@ -42,0 +43,2 @@ pub struct Boundary {\n+    first: u64,\n+    second: u64,\n";
    let old_spans = BTreeMap::from([(
        "rust/runtime/src/example.rs".to_owned(),
        vec![SourceSpan { start: 40, end: 45 }],
    )]);
    let new_spans = BTreeMap::from([(
        "rust/runtime/src/example.rs".to_owned(),
        vec![SourceSpan { start: 40, end: 47 }],
    )]);

    assert!(patch_overlaps_type(patch, &old_spans, &new_spans));
}
