// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Contract tests for secret-safe recorded-agent replay provenance.

use std::collections::BTreeMap;

use aiperf_runtime::graph::replay::{ReplayProvenance, redact_replay_provenance};

#[test]
fn provenance_hashes_cache_namespace_and_redacts_endpoint_credentials() {
    let provenance = ReplayProvenance {
        manifest_digest: "manifest".into(),
        recording_digests: BTreeMap::from([("task".into(), "recording".into())]),
        request_profile_digests: BTreeMap::new(),
        environment_digests: BTreeMap::new(),
        cache_isolation_mode: "first_message_prefix".into(),
        cache_namespace: Some("secret cache namespace".into()),
        cache_namespace_digest: None,
        endpoint: Some("https://user:password@example.test/v1?api_key=secret".into()),
        hardware_description: Some("unknown".into()),
        debug_overrides: Vec::new(),
        comparable: true,
    };
    let redacted = redact_replay_provenance(&provenance);
    let json = serde_json::to_string(&redacted).expect("provenance serializes");
    assert!(!json.contains("secret cache namespace"));
    assert!(!json.contains("password"));
    assert!(!json.contains("api_key"));
    assert_eq!(
        redacted.cache_namespace_digest.as_deref().map(str::len),
        Some(64)
    );
}
