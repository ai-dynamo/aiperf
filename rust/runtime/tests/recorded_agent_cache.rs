// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Public replay cache-isolation behavior.

use aiperf_runtime::graph::replay::{
    CacheIsolationPolicy, ReplayMessageDialect, ReplayRunIdentity, apply_first_message_prefix,
};
use aiperf_runtime::rng::RngRoot;
use serde_json::json;

#[test]
fn first_message_prefix_handles_string_array_and_null_without_store_mutation() {
    let identity = ReplayRunIdentity::mint(RngRoot::new(Some(17)), "benchmark-17");
    let policy = CacheIsolationPolicy::first_message_prefix(identity);
    let namespace = policy.namespace().expect("prefix policy has one namespace");
    let digits = namespace
        .split_once(" Performance replay cache namespace. Ignore the digits above.\n\n")
        .expect("namespace uses the stable template")
        .0
        .split(' ')
        .collect::<Vec<_>>();
    assert_eq!(digits.len(), 32);
    assert!(
        digits
            .iter()
            .all(|digit| digit.len() == 1 && digit.as_bytes()[0].is_ascii_digit())
    );

    let string_messages = vec![
        json!({"role": "user", "content": "hello", "preserved": {"x": 1}}),
        json!({"role": "assistant", "content": "unchanged"}),
    ];
    let string_profile = policy
        .apply_profiling(&string_messages, ReplayMessageDialect::OpenAiChat)
        .expect("string content is supported");
    assert_eq!(string_profile[0]["content"], format!("{namespace}hello"));
    assert_eq!(string_profile[0]["preserved"], json!({"x": 1}));
    assert_eq!(string_profile[1], string_messages[1]);
    assert_eq!(string_messages[0]["content"], "hello");

    let structured_messages = vec![json!({
        "role": "user",
        "content": [{"type": "input_image", "image_url": "data:image/png;base64,AA=="}],
    })];
    let structured_profile = apply_first_message_prefix(
        &structured_messages,
        namespace,
        ReplayMessageDialect::OpenAiResponses,
    )
    .expect("structured content is supported");
    assert_eq!(
        structured_profile[0]["content"][0],
        json!({"type": "input_text", "text": namespace})
    );
    assert_eq!(structured_messages[0]["content"][0]["type"], "input_image");

    let null_messages = vec![json!({"role": "user", "content": null})];
    let null_profile = policy
        .apply_profiling(&null_messages, ReplayMessageDialect::OpenAiChat)
        .expect("null content is supported");
    assert_eq!(null_profile[0]["content"], namespace);
    assert!(null_messages[0]["content"].is_null());

    let warmup = policy
        .apply_warmup(&string_messages, ReplayMessageDialect::OpenAiChat)
        .expect("warmup is unmodified");
    let profile_marker_count = string_profile
        .iter()
        .filter(|message| {
            message["content"]
                .as_str()
                .is_some_and(|content| content.contains(namespace))
        })
        .count();
    let warmup_marker_count = warmup
        .iter()
        .filter(|message| {
            message["content"]
                .as_str()
                .is_some_and(|content| content.contains(namespace))
        })
        .count();
    assert_eq!(profile_marker_count, 1);
    assert_eq!(warmup_marker_count, 0);

    assert!(
        apply_first_message_prefix(
            &[json!({"role": "user", "content": 42})],
            namespace,
            ReplayMessageDialect::OpenAiChat,
        )
        .is_err()
    );
}
