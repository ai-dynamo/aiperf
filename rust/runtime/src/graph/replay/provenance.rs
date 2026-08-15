// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Redacted recorded-agent replay provenance.

use std::collections::BTreeMap;

use serde::Serialize;

/// Controller-collected provenance before publication.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReplayProvenance {
    /// Selected manifest digest.
    pub manifest_digest: String,
    /// Selected decompressed recording digests.
    pub recording_digests: BTreeMap<String, String>,
    /// Resolved request-profile digests.
    pub request_profile_digests: BTreeMap<String, String>,
    /// Resolved environment recipe/image digests.
    pub environment_digests: BTreeMap<String, String>,
    /// Cache isolation mode.
    pub cache_isolation_mode: String,
    /// Protected raw cache namespace retained only in checkpoints.
    pub cache_namespace: Option<String>,
    /// Publication-safe namespace digest when the raw namespace stays protected.
    pub cache_namespace_digest: Option<String>,
    /// Optional endpoint string, potentially containing credentials.
    pub endpoint: Option<String>,
    /// User-provided endpoint hardware description.
    pub hardware_description: Option<String>,
    /// Explicit debug behavior that makes a result non-comparable.
    pub debug_overrides: Vec<String>,
    /// Whether the completed result is comparable.
    pub comparable: bool,
}

/// Publication-safe replay provenance.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RedactedReplayProvenance {
    /// Selected manifest digest.
    pub manifest_digest: String,
    /// Selected decompressed recording digests.
    pub recording_digests: BTreeMap<String, String>,
    /// Resolved request-profile digests.
    pub request_profile_digests: BTreeMap<String, String>,
    /// Resolved environment recipe/image digests.
    pub environment_digests: BTreeMap<String, String>,
    /// Cache isolation mode.
    pub cache_isolation_mode: String,
    /// BLAKE3 digest of the protected namespace, never the namespace itself.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_namespace_digest: Option<String>,
    /// Sanitized endpoint origin and path with userinfo/query/fragment removed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub endpoint: Option<String>,
    /// User-provided endpoint hardware description.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hardware_description: Option<String>,
    /// Explicit debug behavior that makes a result non-comparable.
    pub debug_overrides: Vec<String>,
    /// Whether the completed result is comparable.
    pub comparable: bool,
}

/// Remove credentials and opaque cache bytes before artifact serialization.
#[must_use]
pub fn redact_replay_provenance(provenance: &ReplayProvenance) -> RedactedReplayProvenance {
    RedactedReplayProvenance {
        manifest_digest: provenance.manifest_digest.clone(),
        recording_digests: provenance.recording_digests.clone(),
        request_profile_digests: provenance.request_profile_digests.clone(),
        environment_digests: provenance.environment_digests.clone(),
        cache_isolation_mode: provenance.cache_isolation_mode.clone(),
        cache_namespace_digest: provenance.cache_namespace.as_ref().map_or_else(
            || provenance.cache_namespace_digest.clone(),
            |namespace| Some(blake3::hash(namespace.as_bytes()).to_hex().to_string()),
        ),
        endpoint: provenance.endpoint.as_deref().map(redact_endpoint),
        hardware_description: provenance.hardware_description.clone(),
        debug_overrides: provenance.debug_overrides.clone(),
        comparable: provenance.comparable,
    }
}

fn redact_endpoint(endpoint: &str) -> String {
    let before_query = endpoint.split(['?', '#']).next().unwrap_or_default();
    match before_query.split_once("://") {
        Some((scheme, remainder)) => {
            let host_path = remainder
                .rsplit_once('@')
                .map_or(remainder, |(_, value)| value);
            format!("{scheme}://{host_path}")
        }
        None => before_query.to_string(),
    }
}
