// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Canonical eight-task recorded-agent replay registry and external validation.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use serde::Deserialize;

use super::discovery::{discover_manifest_bytes, parse_manifest_bytes};
use super::{RecordedAgentInputError, RecordedAgentReplayManifest, ValidatedRecordedAgentCorpus};

const CANONICAL_MANIFEST_BYTES: &[u8] =
    include_bytes!("../../../../tests/fixtures/recorded_agent_replay/canonical_manifest.json");
const CANONICAL_DIGEST_INDEX_BYTES: &[u8] =
    include_bytes!("../../../../tests/fixtures/recorded_agent_replay/digest_index.json");
const CANONICAL_MANIFEST_LABEL: &str = "canonical_manifest.json";

/// Digest index pinned beside the canonical replay registry.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CanonicalReplayFixtureDigestIndex {
    /// BLAKE3 digest of canonical manifest bytes.
    pub manifest: String,
    /// BLAKE3 digest of each decompressed recording, keyed by task identity.
    pub recordings: BTreeMap<String, String>,
}

/// Committed canonical replay metadata, independent of a local recording corpus.
#[derive(Clone, Debug)]
pub struct CanonicalReplayFixture {
    /// Strict canonical replay manifest.
    pub manifest: RecordedAgentReplayManifest,
    /// BLAKE3 digest of the committed canonical manifest bytes.
    pub manifest_digest: String,
    /// Committed canonical recording digests.
    pub digest_index: CanonicalReplayFixtureDigestIndex,
}

impl CanonicalReplayFixture {
    /// Load the committed canonical replay metadata without accessing recordings.
    pub fn load() -> Result<Self, RecordedAgentInputError> {
        let label = Path::new(CANONICAL_MANIFEST_LABEL);
        let manifest = parse_manifest_bytes(CANONICAL_MANIFEST_BYTES, label)?;
        let digest_index: CanonicalReplayFixtureDigestIndex =
            serde_json::from_slice(CANONICAL_DIGEST_INDEX_BYTES).map_err(|error| {
                RecordedAgentInputError(format!(
                    "{CANONICAL_MANIFEST_LABEL}: invalid canonical digest index: {error}"
                ))
            })?;
        let manifest_digest = blake3::hash(CANONICAL_MANIFEST_BYTES).to_hex().to_string();
        if manifest_digest != digest_index.manifest {
            return Err(RecordedAgentInputError(format!(
                "{CANONICAL_MANIFEST_LABEL}: canonical manifest digest does not match digest index"
            )));
        }
        let task_keys = manifest
            .tasks
            .iter()
            .map(|task| format!("{}:{}", task.identity.adapter, task.identity.task_id))
            .collect::<BTreeSet<_>>();
        let digest_keys = digest_index
            .recordings
            .keys()
            .cloned()
            .collect::<BTreeSet<_>>();
        if task_keys != digest_keys {
            return Err(RecordedAgentInputError(format!(
                "{CANONICAL_MANIFEST_LABEL}: digest index keys do not match canonical tasks"
            )));
        }
        Ok(Self {
            manifest,
            manifest_digest,
            digest_index,
        })
    }

    /// Validate an explicitly supplied external replay root against this registry.
    pub fn validate_replay_root(
        &self,
        replay_root: &Path,
    ) -> Result<ValidatedRecordedAgentCorpus, RecordedAgentInputError> {
        let corpus = discover_manifest_bytes(
            replay_root,
            CANONICAL_MANIFEST_BYTES,
            Path::new(CANONICAL_MANIFEST_LABEL),
        )?;
        if corpus.manifest_digest.as_deref() != Some(self.manifest_digest.as_str())
            || corpus.recording_digests != self.digest_index.recordings
        {
            return Err(RecordedAgentInputError(format!(
                "{}: external replay root does not match canonical recording digests",
                replay_root.display()
            )));
        }
        Ok(corpus)
    }
}
