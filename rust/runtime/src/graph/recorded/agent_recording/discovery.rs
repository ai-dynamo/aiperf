// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Root-contained decoding and validation for recorded-agent inputs.

use std::collections::{BTreeMap, HashSet};
use std::error::Error;
use std::fmt::{self, Display};
use std::fs;
use std::io::Read;
use std::path::{Component, Path, PathBuf};

use flate2::read::GzDecoder;
use serde_json::Value;

use crate::graph::materialize::decode_additional_body_wire;

use super::schema::{
    ExpectedCorpusShape, RecordedAgentEvent, RecordedAgentRecording, RecordedAgentReplayManifest,
    ReplayTaskIdentity,
};

/// A local source accepted by the recorded-agent input adapter.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RecordedAgentInputSource {
    /// A strict replay manifest.
    Manifest(PathBuf),
    /// One explicit JSON or gzip-compressed recording.
    Recording(PathBuf),
    /// One directory of direct recording children.
    Directory(PathBuf),
}

/// A validation error that names the offending input or source event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecordedAgentInputError(pub String);

impl Display for RecordedAgentInputError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for RecordedAgentInputError {}

/// Decoded recording plus its replay identity and source facts.
#[derive(Clone, Debug)]
pub struct ValidatedRecordedAgentTrace {
    /// Resolved trace identifier.
    pub trace_id: String,
    /// Optional manifest task identity.
    pub identity: Option<ReplayTaskIdentity>,
    /// Resolved recording path.
    pub path: PathBuf,
    /// BLAKE3 digest of decompressed recording bytes.
    pub digest: String,
    /// Resolved source image when one is available.
    pub image: Option<String>,
    /// Validated recording envelope.
    pub recording: RecordedAgentRecording,
}

/// Complete validated source corpus before graph lowering.
#[derive(Clone, Debug)]
pub struct ValidatedRecordedAgentCorpus {
    /// Optional manifest that selected the traces.
    pub manifest: Option<RecordedAgentReplayManifest>,
    /// BLAKE3 digest of raw manifest bytes when a manifest was used.
    pub manifest_digest: Option<String>,
    /// Ordered source traces.
    pub traces: Vec<ValidatedRecordedAgentTrace>,
    /// Recomputed source totals.
    pub shape: ExpectedCorpusShape,
    /// BLAKE3 decompressed-recording digest indexed by task identity or trace id.
    pub recording_digests: BTreeMap<String, String>,
}

/// Discover, decode, and validate a root-contained recorded-agent source.
pub fn discover_recorded_agent_input(
    replay_root: Option<&Path>,
    source: RecordedAgentInputSource,
) -> Result<ValidatedRecordedAgentCorpus, RecordedAgentInputError> {
    let root = replay_root.map(canonical_root).transpose()?;
    match source {
        RecordedAgentInputSource::Manifest(path) => discover_manifest(root.as_deref(), &path),
        RecordedAgentInputSource::Recording(path) => {
            let path = resolve_path(root.as_deref(), &path, "recording source")?;
            let trace = decode_trace(path, None)?;
            finish(None, None, vec![trace])
        }
        RecordedAgentInputSource::Directory(path) => discover_directory(root.as_deref(), &path),
    }
}

fn discover_manifest(
    root: Option<&Path>,
    source: &Path,
) -> Result<ValidatedRecordedAgentCorpus, RecordedAgentInputError> {
    let manifest_path = resolve_path(root, source, "manifest source")?;
    let manifest_bytes = fs::read(&manifest_path).map_err(|error| {
        RecordedAgentInputError(format!("{}: {error}", manifest_path.display()))
    })?;
    let manifest = parse_manifest_bytes(&manifest_bytes, &manifest_path)?;
    let root = root.ok_or_else(|| {
        RecordedAgentInputError(format!(
            "{}: replay_root is required for a replay manifest",
            manifest_path.display()
        ))
    })?;

    discover_manifest_at_root(root, &manifest_bytes, manifest, &manifest_path)
}

/// Discover an explicit replay root selected by trusted manifest bytes.
pub(crate) fn discover_manifest_bytes(
    replay_root: &Path,
    manifest_bytes: &[u8],
    manifest_path: &Path,
) -> Result<ValidatedRecordedAgentCorpus, RecordedAgentInputError> {
    let root = canonical_root(replay_root)?;
    let manifest = parse_manifest_bytes(manifest_bytes, manifest_path)?;
    discover_manifest_at_root(&root, manifest_bytes, manifest, manifest_path)
}

/// Parse and validate strict manifest bytes without resolving any recording path.
pub(crate) fn parse_manifest_bytes(
    bytes: &[u8],
    path: &Path,
) -> Result<RecordedAgentReplayManifest, RecordedAgentInputError> {
    let manifest = serde_json::from_slice(bytes).map_err(|error| {
        RecordedAgentInputError(format!(
            "{}: invalid replay manifest: {error}",
            path.display()
        ))
    })?;
    validate_manifest(&manifest, path)?;
    Ok(manifest)
}

fn discover_manifest_at_root(
    root: &Path,
    manifest_bytes: &[u8],
    manifest: RecordedAgentReplayManifest,
    manifest_path: &Path,
) -> Result<ValidatedRecordedAgentCorpus, RecordedAgentInputError> {
    let mut identities = HashSet::new();
    let mut paths = HashSet::new();
    let mut traces = Vec::with_capacity(manifest.tasks.len());
    for task in &manifest.tasks {
        let identity_key = format!("{}:{}", task.identity.adapter, task.identity.task_id);
        if !identities.insert(identity_key.clone()) {
            return Err(RecordedAgentInputError(format!(
                "{}: duplicate replay task identity {identity_key:?}",
                manifest_path.display()
            )));
        }
        let recording_path = Path::new(&task.recording);
        if recording_path.is_absolute()
            || recording_path
                .components()
                .any(|component| matches!(component, Component::ParentDir))
        {
            return Err(RecordedAgentInputError(format!(
                "{}: recording {:?} escapes replay_root",
                manifest_path.display(),
                task.recording
            )));
        }
        let path = resolve_path(Some(root), recording_path, "manifest recording")?;
        if !paths.insert(path.clone()) {
            return Err(RecordedAgentInputError(format!(
                "{}: duplicate resolved recording path {}",
                manifest_path.display(),
                path.display()
            )));
        }
        traces.push(decode_trace(path, Some(task.identity.clone()))?);
    }
    let manifest_digest = blake3::hash(manifest_bytes).to_hex().to_string();
    finish(Some(manifest), Some(manifest_digest), traces)
}

fn discover_directory(
    root: Option<&Path>,
    source: &Path,
) -> Result<ValidatedRecordedAgentCorpus, RecordedAgentInputError> {
    let directory = resolve_path(root, source, "recording directory")?;
    if !directory.is_dir() {
        return Err(RecordedAgentInputError(format!(
            "{}: recorded-agent directory source is not a directory",
            directory.display()
        )));
    }
    let mut entries = fs::read_dir(&directory)
        .map_err(|error| RecordedAgentInputError(format!("{}: {error}", directory.display())))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| RecordedAgentInputError(format!("{}: {error}", directory.display())))?;
    entries.sort_by_key(|entry| entry.file_name());

    let mut traces = Vec::new();
    for entry in entries {
        let path = entry.path();
        let metadata = fs::symlink_metadata(&path)
            .map_err(|error| RecordedAgentInputError(format!("{}: {error}", path.display())))?;
        if metadata.file_type().is_symlink() {
            return Err(RecordedAgentInputError(format!(
                "{}: symlink inputs are forbidden beneath replay_root",
                path.display()
            )));
        }
        if !metadata.is_file() || !is_recording_name(&path) {
            continue;
        }
        let bytes = decode_bytes(&path)?;
        let value: Value = serde_json::from_slice(&bytes).map_err(|error| {
            RecordedAgentInputError(format!(
                "{}: invalid recording JSON: {error}",
                path.display()
            ))
        })?;
        if !is_recording_value(&value) {
            continue;
        }
        traces.push(decode_trace_from_bytes(path, bytes, None)?);
    }
    finish(None, None, traces)
}

fn finish(
    manifest: Option<RecordedAgentReplayManifest>,
    manifest_digest: Option<String>,
    traces: Vec<ValidatedRecordedAgentTrace>,
) -> Result<ValidatedRecordedAgentCorpus, RecordedAgentInputError> {
    if traces.is_empty() {
        return Err(RecordedAgentInputError(
            "recorded-agent input contains no recordings".into(),
        ));
    }
    let mut trace_ids = HashSet::new();
    let mut recording_digests = BTreeMap::new();
    let mut shape = ExpectedCorpusShape {
        total_isl: 0,
        isl_delta: 0,
        peak_isl: 0,
        total_osl: 0,
        model_calls: 0,
        tool_calls: 0,
        tool_duration_ms: 0.0,
        max_tool_call_duration_ms: 0.0,
        timed_out_tool_calls: 0,
    };
    for trace in &traces {
        if !trace_ids.insert(trace.trace_id.clone()) {
            return Err(RecordedAgentInputError(format!(
                "{}: duplicate resolved trace id {:?}",
                trace.path.display(),
                trace.trace_id
            )));
        }
        let digest_key = trace.identity.as_ref().map_or_else(
            || trace.trace_id.clone(),
            |identity| format!("{}:{}", identity.adapter, identity.task_id),
        );
        recording_digests.insert(digest_key, trace.digest.clone());
        merge_shape(&mut shape, trace)?;
    }
    if let Some(manifest) = &manifest {
        compare_shape(&manifest.aggregate, &shape)?;
    }
    Ok(ValidatedRecordedAgentCorpus {
        manifest,
        manifest_digest,
        traces,
        shape,
        recording_digests,
    })
}

fn decode_trace(
    path: PathBuf,
    identity: Option<ReplayTaskIdentity>,
) -> Result<ValidatedRecordedAgentTrace, RecordedAgentInputError> {
    let bytes = decode_bytes(&path)?;
    decode_trace_from_bytes(path, bytes, identity)
}

fn decode_trace_from_bytes(
    path: PathBuf,
    bytes: Vec<u8>,
    identity: Option<ReplayTaskIdentity>,
) -> Result<ValidatedRecordedAgentTrace, RecordedAgentInputError> {
    let recording: RecordedAgentRecording = serde_json::from_slice(&bytes).map_err(|error| {
        RecordedAgentInputError(format!(
            "{}: invalid recording JSON: {error}",
            path.display()
        ))
    })?;
    if !recording.format.starts_with("mini-swe-agent-recording-") {
        return Err(RecordedAgentInputError(format!(
            "{}: unsupported recording format {:?}",
            path.display(),
            recording.format
        )));
    }
    let trace_id = resolve_trace_id(&recording, &path).ok_or_else(|| {
        RecordedAgentInputError(format!("{}: recording has no trace id", path.display()))
    })?;
    let image = resolve_image(&recording);
    validate_events(&recording, &trace_id, &path)?;
    Ok(ValidatedRecordedAgentTrace {
        trace_id,
        identity,
        path,
        digest: blake3::hash(&bytes).to_hex().to_string(),
        image,
        recording,
    })
}

fn decode_bytes(path: &Path) -> Result<Vec<u8>, RecordedAgentInputError> {
    let file = fs::File::open(path)
        .map_err(|error| RecordedAgentInputError(format!("{}: {error}", path.display())))?;
    if path.extension().is_some_and(|extension| extension == "gz") {
        let mut bytes = Vec::new();
        GzDecoder::new(file)
            .read_to_end(&mut bytes)
            .map_err(|error| {
                RecordedAgentInputError(format!(
                    "{}: invalid gzip recording: {error}",
                    path.display()
                ))
            })?;
        Ok(bytes)
    } else {
        fs::read(path)
            .map_err(|error| RecordedAgentInputError(format!("{}: {error}", path.display())))
    }
}

fn validate_manifest(
    manifest: &RecordedAgentReplayManifest,
    path: &Path,
) -> Result<(), RecordedAgentInputError> {
    if manifest.name.trim().is_empty() || manifest.mode != "replay" || manifest.tasks.is_empty() {
        return Err(RecordedAgentInputError(format!(
            "{}: replay manifest requires non-empty name and tasks with mode replay",
            path.display()
        )));
    }
    let defaults = &manifest.defaults;
    if defaults.config.trim().is_empty()
        || defaults.step_limit == 0
        || !defaults.cost_limit.is_finite()
        || defaults.cost_limit < 0.0
        || defaults.environment_class.trim().is_empty()
        || defaults.docker_network != "none"
        || !defaults.per_inference_timeout.is_finite()
        || defaults.per_inference_timeout <= 0.0
        || defaults.fallback_max_output_tokens == 0
        || !defaults.temperature.is_finite()
        || defaults.temperature < 0.0
        || !defaults.top_p.is_finite()
        || !(0.0..=1.0).contains(&defaults.top_p)
        || defaults.top_k == 0
        || !defaults.min_p.is_finite()
        || !(0.0..=1.0).contains(&defaults.min_p)
        || defaults.measurement_scope != "agent_run_only"
    {
        return Err(RecordedAgentInputError(format!(
            "{}: replay manifest has invalid strict defaults",
            path.display()
        )));
    }
    decode_additional_body_wire(
        defaults.extra_request_body.get().as_bytes(),
        "manifest extra_request_body",
    )
    .map_err(|error| RecordedAgentInputError(format!("{}: {error}", path.display())))?;
    for task in &manifest.tasks {
        if !matches!(task.identity.adapter.as_str(), "pinchbench" | "swebench")
            || task.identity.family.trim().is_empty()
            || task.identity.task_id.trim().is_empty()
            || task.recording.trim().is_empty()
        {
            return Err(RecordedAgentInputError(format!(
                "{}: replay manifest has invalid task {:?}",
                path.display(),
                task.identity.task_id
            )));
        }
    }
    validate_shape(&manifest.aggregate, path)
}

fn validate_shape(shape: &ExpectedCorpusShape, path: &Path) -> Result<(), RecordedAgentInputError> {
    if !shape.tool_duration_ms.is_finite()
        || shape.tool_duration_ms < 0.0
        || !shape.max_tool_call_duration_ms.is_finite()
        || shape.max_tool_call_duration_ms < 0.0
    {
        return Err(RecordedAgentInputError(format!(
            "{}: replay manifest has non-finite source shape duration",
            path.display()
        )));
    }
    Ok(())
}

fn validate_events(
    recording: &RecordedAgentRecording,
    trace_id: &str,
    path: &Path,
) -> Result<(), RecordedAgentInputError> {
    for event in &recording.events {
        if !event.timestamp.is_finite() || event.timestamp <= 0.0 {
            return Err(event_error(
                path,
                trace_id,
                event,
                "timestamp must be positive and finite",
            ));
        }
        if event.event_type == "model_call" {
            if event.error.is_some() {
                return Err(event_error(path, trace_id, event, "model call failed"));
            }
            if event
                .provider_request
                .as_ref()
                .and_then(|request| request.messages.as_ref())
                .is_none()
            {
                return Err(event_error(
                    path,
                    trace_id,
                    event,
                    "model call lacks provider_request.messages",
                ));
            }
        }
    }
    Ok(())
}

fn merge_shape(
    shape: &mut ExpectedCorpusShape,
    trace: &ValidatedRecordedAgentTrace,
) -> Result<(), RecordedAgentInputError> {
    let mut previous_isl = None;
    for event in &trace.recording.events {
        if event.event_type == "model_call" {
            let (isl, osl) = response_usage(event);
            shape.total_isl = shape.total_isl.saturating_add(isl);
            shape.total_osl = shape.total_osl.saturating_add(osl);
            shape.model_calls = shape.model_calls.saturating_add(1);
            shape.peak_isl = shape.peak_isl.max(isl);
            shape.isl_delta = shape.isl_delta.saturating_add(match previous_isl {
                Some(previous) => isl.saturating_sub(previous),
                None => isl,
            });
            previous_isl = Some(isl);
        }
        if is_eligible_tool_call(event) {
            let duration_ms = event.duration_ns.unwrap_or_default() as f64 / 1_000_000.0;
            shape.tool_calls = shape.tool_calls.saturating_add(1);
            shape.tool_duration_ms += duration_ms;
            shape.max_tool_call_duration_ms = shape.max_tool_call_duration_ms.max(duration_ms);
        } else if is_tool_timeout(event) {
            shape.timed_out_tool_calls = shape.timed_out_tool_calls.saturating_add(1);
        }
    }
    if !shape.tool_duration_ms.is_finite() || !shape.max_tool_call_duration_ms.is_finite() {
        return Err(RecordedAgentInputError(format!(
            "{}: source shape duration is non-finite",
            trace.path.display()
        )));
    }
    Ok(())
}

fn response_usage(event: &RecordedAgentEvent) -> (u64, u64) {
    let usage = event
        .response_message
        .as_ref()
        .and_then(|response| response.pointer("/extra/response/usage"));
    let value = |field| {
        usage
            .and_then(|usage| usage.get(field))
            .and_then(Value::as_u64)
            .unwrap_or(0)
    };
    (value("prompt_tokens"), value("completion_tokens"))
}

fn is_eligible_tool_call(event: &RecordedAgentEvent) -> bool {
    event.event_type == "tool_call"
        && event
            .action
            .as_ref()
            .and_then(|action| action.get("command"))
            .and_then(Value::as_str)
            .is_some()
        && event.error.as_ref().is_none_or(is_completed_control_flow)
}

fn is_completed_control_flow(error: &Value) -> bool {
    error
        .get("type")
        .and_then(Value::as_str)
        .is_some_and(|kind| {
            matches!(
                kind,
                "InterruptAgentFlow"
                    | "Submitted"
                    | "LimitsExceeded"
                    | "ReplayExhausted"
                    | "UserInterruption"
                    | "FormatError"
            )
        })
}

fn is_tool_timeout(event: &RecordedAgentEvent) -> bool {
    event.event_type == "tool_call"
        && event.error.as_ref().is_some_and(|error| {
            error
                .get("type")
                .and_then(Value::as_str)
                .is_some_and(|kind| kind.to_ascii_lowercase().contains("timeout"))
        })
}

fn compare_shape(
    expected: &ExpectedCorpusShape,
    actual: &ExpectedCorpusShape,
) -> Result<(), RecordedAgentInputError> {
    if expected.total_isl != actual.total_isl
        || expected.isl_delta != actual.isl_delta
        || expected.peak_isl != actual.peak_isl
        || expected.total_osl != actual.total_osl
        || expected.model_calls != actual.model_calls
        || expected.tool_calls != actual.tool_calls
        || expected.timed_out_tool_calls != actual.timed_out_tool_calls
        || (expected.tool_duration_ms - actual.tool_duration_ms).abs() > 1e-6
        || (expected.max_tool_call_duration_ms - actual.max_tool_call_duration_ms).abs() > 1e-6
    {
        return Err(RecordedAgentInputError(format!(
            "replay manifest aggregate differs from recomputed source shape: expected {expected:?}, got {actual:?}"
        )));
    }
    Ok(())
}

fn resolve_trace_id(recording: &RecordedAgentRecording, path: &Path) -> Option<String> {
    let instance_id = recording.metadata.instance.as_ref().and_then(|instance| {
        instance
            .get("instance_id")
            .and_then(Value::as_str)
            .map(str::to_owned)
    });
    [
        recording.metadata.instance_id.clone(),
        recording.metadata.task_id.clone(),
        instance_id,
        file_stem(path),
    ]
    .into_iter()
    .flatten()
    .find(|value| !value.trim().is_empty())
}

fn resolve_image(recording: &RecordedAgentRecording) -> Option<String> {
    let instance = recording.metadata.instance.as_ref();
    [
        recording.metadata.docker_image.clone(),
        instance
            .and_then(|instance| instance.get("image_name"))
            .and_then(Value::as_str)
            .map(str::to_owned),
        instance
            .and_then(|instance| instance.get("docker_image"))
            .and_then(Value::as_str)
            .map(str::to_owned),
        instance
            .and_then(|instance| instance.get("instance_id"))
            .and_then(Value::as_str)
            .map(|id| format!("docker.io/swebench/sweb.eval.x86_64.{id}:latest")),
    ]
    .into_iter()
    .flatten()
    .find(|value| !value.trim().is_empty())
}

fn file_stem(path: &Path) -> Option<String> {
    let name = path.file_name()?.to_str()?;
    let name = name.strip_suffix(".gz").unwrap_or(name);
    name.strip_suffix(".json").map(str::to_owned)
}

fn event_error(
    path: &Path,
    trace_id: &str,
    event: &RecordedAgentEvent,
    detail: &str,
) -> RecordedAgentInputError {
    RecordedAgentInputError(format!(
        "{}: trace {:?} event {} type {:?}: {detail}",
        path.display(),
        trace_id,
        event.id,
        event.event_type
    ))
}

fn canonical_root(root: &Path) -> Result<PathBuf, RecordedAgentInputError> {
    let root = fs::canonicalize(root).map_err(|error| {
        RecordedAgentInputError(format!("{}: replay_root: {error}", root.display()))
    })?;
    if !root.is_dir() {
        return Err(RecordedAgentInputError(format!(
            "{}: replay_root is not a directory",
            root.display()
        )));
    }
    Ok(root)
}

fn resolve_path(
    root: Option<&Path>,
    path: &Path,
    label: &str,
) -> Result<PathBuf, RecordedAgentInputError> {
    let requested = if path.is_absolute() {
        path.to_path_buf()
    } else if let Some(root) = root {
        root.join(path)
    } else {
        path.to_path_buf()
    };
    reject_symlink_components(&requested)?;
    let canonical = fs::canonicalize(&requested).map_err(|error| {
        RecordedAgentInputError(format!("{}: {label}: {error}", requested.display()))
    })?;
    if let Some(root) = root
        && !canonical.starts_with(root)
    {
        return Err(RecordedAgentInputError(format!(
            "{}: {label} escapes replay_root {}",
            requested.display(),
            root.display()
        )));
    }
    Ok(canonical)
}

fn reject_symlink_components(path: &Path) -> Result<(), RecordedAgentInputError> {
    let mut current = PathBuf::new();
    for component in path.components() {
        current.push(component.as_os_str());
        if let Ok(metadata) = fs::symlink_metadata(&current)
            && metadata.file_type().is_symlink()
        {
            return Err(RecordedAgentInputError(format!(
                "{}: symlink inputs are forbidden beneath replay_root",
                current.display()
            )));
        }
    }
    Ok(())
}

fn is_recording_name(path: &Path) -> bool {
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    name.ends_with(".json") || name.ends_with(".json.gz")
}

fn is_recording_value(value: &Value) -> bool {
    value
        .get("format")
        .and_then(Value::as_str)
        .is_some_and(|format| format.starts_with("mini-swe-agent-recording-"))
}
