// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Controller-owned atomic checkpointing for recorded-agent replay.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt::{self, Display};
use std::fs;
use std::io::Write;
use std::path::Path;

#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;

use serde::de::{self, Deserializer, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Serialize};

use crate::graph::driver::ReplayTaskIdentity;

use super::ReplayRunIdentity;

/// Bounded terminal classification retained in a replay checkpoint.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReplayTaskClassification {
    /// Profiling completed and all expected calls were observed.
    Successful,
    /// The trace reached a terminal failure.
    Failed,
    /// The trace ended without enough evidence to verify completion.
    Partial,
}

/// Controller-folded terminal facts for one replay task.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CompletedReplayTask {
    /// Source manifest ordinal.
    pub manifest_ordinal: usize,
    /// Source recording digest.
    pub source_digest: String,
    /// Resolved request-profile digest.
    pub request_profile_digest: String,
    /// Resolved environment recipe digest.
    pub environment_digest: String,
    /// Successful profiling model-call count.
    pub successful_call_count: u64,
    /// Terminal task classification.
    pub classification: ReplayTaskClassification,
    /// Inclusive per-record artifact offset at task start.
    #[serde(default)]
    pub artifact_offset_start: u64,
    /// Exclusive per-record artifact offset at task end.
    #[serde(default)]
    pub artifact_offset_end: u64,
}

impl CompletedReplayTask {
    /// Build verified successful terminal facts.
    #[must_use]
    pub fn successful(
        manifest_ordinal: usize,
        source_digest: impl Into<String>,
        request_profile_digest: impl Into<String>,
        environment_digest: impl Into<String>,
        successful_call_count: u64,
    ) -> Self {
        Self {
            manifest_ordinal,
            source_digest: source_digest.into(),
            request_profile_digest: request_profile_digest.into(),
            environment_digest: environment_digest.into(),
            successful_call_count,
            classification: ReplayTaskClassification::Successful,
            artifact_offset_start: 0,
            artifact_offset_end: 0,
        }
    }

    /// Build an incomplete terminal fact which must always rerun on resume.
    #[must_use]
    pub fn partial(
        manifest_ordinal: usize,
        source_digest: impl Into<String>,
        request_profile_digest: impl Into<String>,
        environment_digest: impl Into<String>,
        successful_call_count: u64,
    ) -> Self {
        Self {
            classification: ReplayTaskClassification::Partial,
            ..Self::successful(
                manifest_ordinal,
                source_digest,
                request_profile_digest,
                environment_digest,
                successful_call_count,
            )
        }
    }

    /// Whether this terminal fact is sufficient to skip a task on resume.
    #[must_use]
    pub fn is_verified_success_for(
        &self,
        _task: &ReplayTaskIdentity,
        manifest_ordinal: usize,
        source_digest: &str,
        request_profile_digest: &str,
        expected_call_count: u64,
    ) -> bool {
        self.classification == ReplayTaskClassification::Successful
            && self.manifest_ordinal == manifest_ordinal
            && self.source_digest == source_digest
            && self.request_profile_digest == request_profile_digest
            && self.successful_call_count == expected_call_count
    }
}

/// Persistent controller-owned replay progress.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReplayCheckpoint {
    /// Opaque replay-run identity and protected namespace state.
    pub run: ReplayRunIdentity,
    /// Manifest digest copied to the checkpoint header for fail-closed loading.
    pub manifest_digest: String,
    /// Completed bounded terminal facts keyed by stable task identity.
    pub completed: BTreeMap<ReplayTaskIdentity, CompletedReplayTask>,
}

impl ReplayCheckpoint {
    /// Start an empty checkpoint before warmup or profiling begins.
    #[must_use]
    pub fn new(run: ReplayRunIdentity, manifest_digest: impl Into<String>) -> Self {
        Self {
            run,
            manifest_digest: manifest_digest.into(),
            completed: BTreeMap::new(),
        }
    }

    /// Add one completed task for test and single-owner controller composition.
    #[must_use]
    pub fn with_completed(
        mut self,
        task: ReplayTaskIdentity,
        completed: CompletedReplayTask,
    ) -> Self {
        self.completed.insert(task, completed);
        self
    }

    /// Persist the complete checkpoint by atomic replacement.
    pub fn write_atomic(&self, path: &Path) -> Result<(), ReplayResumeError> {
        let identity = self.run.checkpoint_identity().ok_or_else(|| {
            ReplayResumeError::new("checkpoint identity is missing persistent replay fields")
        })?;
        let disk = ReplayCheckpointDisk {
            version: 2,
            run_id: identity.run_id.clone(),
            replay_root_digest: identity.replay_root_digest.clone(),
            manifest_digest: identity.manifest_digest.clone(),
            recording_digests: identity.recording_digests.clone(),
            request_profile_digests: identity.request_profile_digests.clone(),
            environment_digests: identity.environment_digests.clone(),
            cache_namespace: identity.cache_namespace.clone(),
            completed: CheckpointTasks(
                self.completed
                    .iter()
                    .map(|(identity, completed)| CheckpointTask {
                        identity: identity.clone(),
                        completed: completed.clone(),
                    })
                    .collect(),
            ),
        };
        let bytes = serde_json::to_vec(&disk).map_err(|error| {
            ReplayResumeError::new(format!("serializing replay checkpoint: {error}"))
        })?;
        let parent = path
            .parent()
            .ok_or_else(|| ReplayResumeError::new("checkpoint path has no parent"))?;
        fs::create_dir_all(parent).map_err(|error| {
            ReplayResumeError::new(format!("creating checkpoint directory: {error}"))
        })?;
        let temporary = path.with_extension("tmp");
        let mut file = fs::File::create(&temporary).map_err(|error| {
            ReplayResumeError::new(format!("creating replay checkpoint: {error}"))
        })?;
        #[cfg(unix)]
        file.set_permissions(fs::Permissions::from_mode(0o600))
            .map_err(|error| {
                ReplayResumeError::new(format!("protecting replay checkpoint: {error}"))
            })?;
        file.write_all(&bytes).map_err(|error| {
            ReplayResumeError::new(format!("writing replay checkpoint: {error}"))
        })?;
        file.sync_all().map_err(|error| {
            ReplayResumeError::new(format!("syncing replay checkpoint: {error}"))
        })?;
        fs::rename(&temporary, path).map_err(|error| {
            ReplayResumeError::new(format!("replacing replay checkpoint: {error}"))
        })?;
        fs::File::open(parent)
            .and_then(|directory| directory.sync_all())
            .map_err(|error| {
                ReplayResumeError::new(format!("syncing replay checkpoint directory: {error}"))
            })
    }

    /// Load a checkpoint and reject mismatched persistent identity.
    pub fn read_for_resume(
        path: &Path,
        run: &ReplayRunIdentity,
    ) -> Result<Self, ReplayResumeError> {
        let bytes = fs::read(path).map_err(|error| {
            ReplayResumeError::new(format!("reading replay checkpoint: {error}"))
        })?;
        let disk: ReplayCheckpointDisk = serde_json::from_slice(&bytes).map_err(|error| {
            ReplayResumeError::new(format!("parsing replay checkpoint: {error}"))
        })?;
        if disk.version > 2 {
            return Err(ReplayResumeError::new(format!(
                "unsupported replay checkpoint schema version {}",
                disk.version
            )));
        }
        let checkpoint = Self {
            run: ReplayRunIdentity::for_checkpoint_with_environment(
                disk.run_id,
                disk.replay_root_digest,
                disk.manifest_digest.clone(),
                disk.recording_digests,
                disk.request_profile_digests,
                disk.environment_digests,
                disk.cache_namespace,
            ),
            manifest_digest: disk.manifest_digest,
            completed: completed_map(disk.completed.0)?,
        };
        checkpoint.validate_resume(run)?;
        Ok(checkpoint)
    }

    /// Recover the protected namespace from disk before validating a resumed run.
    ///
    /// A second unseeded invocation cannot reproduce the namespace by minting it
    /// again, so the controller must build its comparison identity from this
    /// protected checkpoint value.
    pub fn restore_run_identity(
        path: &Path,
        run_id: impl Into<String>,
        replay_root_digest: impl Into<String>,
        manifest_digest: impl Into<String>,
        recording_digests: BTreeMap<String, String>,
        request_profile_digests: BTreeMap<String, String>,
        environment_digests: BTreeMap<String, String>,
    ) -> Result<ReplayRunIdentity, ReplayResumeError> {
        let checkpoint = Self::read(path)?;
        let identity = checkpoint.run.checkpoint_identity().ok_or_else(|| {
            ReplayResumeError::IdentityMismatch("checkpoint has no persistent identity".into())
        })?;
        Ok(ReplayRunIdentity::for_checkpoint_with_environment(
            run_id,
            replay_root_digest,
            manifest_digest,
            recording_digests,
            request_profile_digests,
            environment_digests,
            identity.cache_namespace.clone(),
        ))
    }

    fn read(path: &Path) -> Result<Self, ReplayResumeError> {
        let bytes = fs::read(path).map_err(|error| {
            ReplayResumeError::new(format!("reading replay checkpoint: {error}"))
        })?;
        let disk: ReplayCheckpointDisk = serde_json::from_slice(&bytes).map_err(|error| {
            ReplayResumeError::new(format!("parsing replay checkpoint: {error}"))
        })?;
        if disk.version > 2 {
            return Err(ReplayResumeError::new(format!(
                "unsupported replay checkpoint schema version {}",
                disk.version
            )));
        }
        Ok(Self {
            run: ReplayRunIdentity::for_checkpoint_with_environment(
                disk.run_id,
                disk.replay_root_digest,
                disk.manifest_digest.clone(),
                disk.recording_digests,
                disk.request_profile_digests,
                disk.environment_digests,
                disk.cache_namespace,
            ),
            manifest_digest: disk.manifest_digest,
            completed: completed_map(disk.completed.0)?,
        })
    }

    /// Reject any resume attempt with a changed root, source, profile, or namespace.
    pub fn validate_resume(&self, run: &ReplayRunIdentity) -> Result<(), ReplayResumeError> {
        let Some(saved) = self.run.checkpoint_identity() else {
            return Err(ReplayResumeError::IdentityMismatch(
                "checkpoint has no persistent identity".into(),
            ));
        };
        let Some(current) = run.checkpoint_identity() else {
            return Err(ReplayResumeError::IdentityMismatch(
                "resume run has no persistent identity".into(),
            ));
        };
        if saved.replay_root_digest != current.replay_root_digest
            || saved.manifest_digest != current.manifest_digest
            || saved.recording_digests != current.recording_digests
            || saved.request_profile_digests != current.request_profile_digests
            || saved.environment_digests != current.environment_digests
            || saved.cache_namespace != current.cache_namespace
            || self.manifest_digest != current.manifest_digest
        {
            return Err(ReplayResumeError::IdentityMismatch(
                "recorded-agent replay identity changed".into(),
            ));
        }
        Ok(())
    }

    /// Return true only for an exact, successful, fully verified prior task.
    #[must_use]
    pub fn should_skip(
        &self,
        task: &ReplayTaskIdentity,
        manifest_ordinal: usize,
        source_digest: &str,
        request_profile_digest: &str,
        expected_call_count: u64,
    ) -> bool {
        self.completed.get(task).is_some_and(|completed| {
            completed.is_verified_success_for(
                task,
                manifest_ordinal,
                source_digest,
                request_profile_digest,
                expected_call_count,
            )
        })
    }
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ReplayCheckpointDisk {
    #[serde(default = "checkpoint_schema_v1")]
    version: u8,
    run_id: String,
    replay_root_digest: String,
    manifest_digest: String,
    recording_digests: BTreeMap<String, String>,
    request_profile_digests: BTreeMap<String, String>,
    #[serde(default)]
    environment_digests: BTreeMap<String, String>,
    cache_namespace: String,
    completed: CheckpointTasks,
}

fn checkpoint_schema_v1() -> u8 {
    1
}

fn completed_map(
    completed: Vec<CheckpointTask>,
) -> Result<BTreeMap<ReplayTaskIdentity, CompletedReplayTask>, ReplayResumeError> {
    let mut entries = BTreeMap::new();
    for entry in completed {
        if entries
            .insert(entry.identity.clone(), entry.completed)
            .is_some()
        {
            return Err(ReplayResumeError::new(format!(
                "replay checkpoint contains duplicate task identity {}:{}",
                entry.identity.adapter, entry.identity.task_id
            )));
        }
    }
    Ok(entries)
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct CheckpointTask {
    identity: ReplayTaskIdentity,
    completed: CompletedReplayTask,
}

/// Current vector checkpoint entries with a reader for the historical map form.
#[derive(Serialize)]
struct CheckpointTasks(Vec<CheckpointTask>);

impl<'de> Deserialize<'de> for CheckpointTasks {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct CheckpointTasksVisitor;

        impl<'de> Visitor<'de> for CheckpointTasksVisitor {
            type Value = CheckpointTasks;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a vector of checkpoint tasks or a legacy task map")
            }

            fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut tasks = Vec::new();
                while let Some(task) = sequence.next_element::<CheckpointTask>()? {
                    tasks.push(task);
                }
                Ok(CheckpointTasks(tasks))
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut tasks = Vec::new();
                let mut keys = BTreeMap::new();
                while let Some((key, completed)) =
                    map.next_entry::<String, CompletedReplayTask>()?
                {
                    if keys.insert(key.clone(), ()).is_some() {
                        return Err(de::Error::custom(format!(
                            "replay checkpoint legacy task map contains duplicate key {key:?}"
                        )));
                    }
                    let identity = serde_json::from_str(&key).map_err(|error| {
                        de::Error::custom(format!(
                            "replay checkpoint legacy task key must be a JSON replay identity: {error}"
                        ))
                    })?;
                    tasks.push(CheckpointTask {
                        identity,
                        completed,
                    });
                }
                Ok(CheckpointTasks(tasks))
            }
        }

        deserializer.deserialize_any(CheckpointTasksVisitor)
    }
}

/// Failure loading, writing, or validating persistent replay state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReplayResumeError {
    /// Persisted replay identity does not match the proposed resume run.
    IdentityMismatch(String),
    /// A checkpoint filesystem or encoding operation failed.
    Other(String),
}

impl ReplayResumeError {
    fn new(message: impl Into<String>) -> Self {
        Self::Other(message.into())
    }
}

impl Display for ReplayResumeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IdentityMismatch(message) | Self::Other(message) => formatter.write_str(message),
        }
    }
}

impl Error for ReplayResumeError {}
