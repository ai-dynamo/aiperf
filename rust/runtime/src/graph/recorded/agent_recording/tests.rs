// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tests for strict recorded-agent input discovery.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use flate2::Compression;
use flate2::write::GzEncoder;
use serde_json::json;
use tempfile::TempDir;

use super::{CanonicalReplayFixture, RecordedAgentInputSource, discover_recorded_agent_input};

fn root() -> TempDir {
    tempfile::tempdir().unwrap()
}

fn recording(
    trace_id: &str,
    image: Option<&str>,
    command_error: Option<&str>,
) -> serde_json::Value {
    let error = command_error.map(|kind| json!({"type": kind, "message": "terminal"}));
    json!({
        "format": "mini-swe-agent-recording-1.0",
        "metadata": {
            "instance_id": trace_id,
            "docker_image": image,
            "instance": {"instance_id": "fallback", "image_name": "nested-image"}
        },
        "events": [
            {
                "id": 1,
                "type": "model_call",
                "timestamp": 1.0,
                "duration_ns": 10,
                "provider_request": {"messages": [{"role": "user", "content": "one"}]},
                "response_message": {"extra": {"response": {"usage": {"prompt_tokens": 3, "completion_tokens": 2}}}}
            },
            {
                "id": 2,
                "type": "model_call",
                "timestamp": 2.0,
                "duration_ns": 11,
                "provider_request": {"messages": [{"role": "user", "content": "two"}]},
                "response_message": {"extra": {"response": {"usage": {"prompt_tokens": 5, "completion_tokens": 4}}}}
            },
            {
                "id": 3,
                "type": "tool_call",
                "timestamp": 3.0,
                "duration_ns": 4_000_000,
                "action": {"command": "true"},
                "error": error
            }
        ]
    })
}

fn write_json(path: &Path, value: &serde_json::Value) {
    fs::write(path, serde_json::to_vec(value).unwrap()).unwrap();
}

fn write_gzip(path: &Path, value: &serde_json::Value) {
    let mut encoder = GzEncoder::new(fs::File::create(path).unwrap(), Compression::default());
    encoder
        .write_all(&serde_json::to_vec(value).unwrap())
        .unwrap();
    encoder.finish().unwrap();
}

#[test]
fn manifest_rejects_root_escape_and_duplicate_task_identity() {
    let root = root();
    write_json(
        &root.path().join("inside.json"),
        &recording("inside", None, None),
    );
    let manifest = json!({
        "name": "test", "mode": "replay", "defaults": defaults(), "aggregate": shape(),
        "attribution": {"source": "test"},
        "tasks": [
            task("one", "../escape.json"),
            task("one", "inside.json")
        ]
    });
    write_json(&root.path().join("manifest.json"), &manifest);

    let error = discover_recorded_agent_input(
        Some(root.path()),
        RecordedAgentInputSource::Manifest(PathBuf::from("manifest.json")),
    )
    .unwrap_err();

    assert!(error.to_string().contains("replay_root"));
}

#[test]
fn manifest_rejects_duplicate_task_identity() {
    let root = root();
    write_json(
        &root.path().join("inside.json"),
        &recording("inside", None, None),
    );
    let manifest = json!({
        "name": "test", "mode": "replay", "defaults": defaults(), "aggregate": shape(),
        "attribution": {"source": "test"},
        "tasks": [task("one", "inside.json"), task("one", "inside.json")]
    });
    write_json(&root.path().join("manifest.json"), &manifest);

    let error = discover_recorded_agent_input(
        Some(root.path()),
        RecordedAgentInputSource::Manifest(PathBuf::from("manifest.json")),
    )
    .unwrap_err();

    assert!(error.to_string().contains("duplicate replay task identity"));
}

#[test]
fn shallow_directory_is_sorted_and_gzip_is_decoded() {
    let root = root();
    let recordings = root.path().join("recordings");
    fs::create_dir(&recordings).unwrap();
    write_gzip(&recordings.join("b.json.gz"), &recording("b", None, None));
    write_json(&recordings.join("a.json"), &recording("a", None, None));
    fs::create_dir(recordings.join("nested")).unwrap();
    write_json(
        &recordings.join("nested").join("ignored.json"),
        &recording("nested", None, None),
    );
    write_json(
        &recordings.join("manifest.json"),
        &json!({"name": "not a recording"}),
    );

    let corpus = discover_recorded_agent_input(
        Some(root.path()),
        RecordedAgentInputSource::Directory(PathBuf::from("recordings")),
    )
    .unwrap();

    assert_eq!(
        corpus
            .traces
            .iter()
            .map(|trace| trace.trace_id.as_str())
            .collect::<Vec<_>>(),
        vec!["a", "b"]
    );
    assert_eq!(corpus.shape.total_isl, 16);
    assert_eq!(corpus.shape.isl_delta, 10);
    assert_eq!(corpus.shape.peak_isl, 5);
    assert_eq!(corpus.shape.total_osl, 12);
    assert_eq!(corpus.shape.model_calls, 4);
    assert_eq!(corpus.shape.tool_calls, 2);
    assert_eq!(corpus.shape.tool_duration_ms, 8.0);
    assert_eq!(corpus.shape.max_tool_call_duration_ms, 4.0);
    assert_eq!(corpus.shape.timed_out_tool_calls, 0);
}

#[test]
fn trace_identity_image_precedence_and_completed_control_flow_are_preserved() {
    let root = root();
    write_json(
        &root.path().join("recording.json"),
        &recording("preferred-id", Some("preferred-image"), Some("Submitted")),
    );

    let corpus = discover_recorded_agent_input(
        Some(root.path()),
        RecordedAgentInputSource::Recording(PathBuf::from("recording.json")),
    )
    .unwrap();

    assert_eq!(corpus.traces[0].trace_id, "preferred-id");
    assert_eq!(corpus.traces[0].image.as_deref(), Some("preferred-image"));
    assert_eq!(corpus.shape.tool_calls, 1);
    assert_eq!(corpus.shape.timed_out_tool_calls, 0);
}

#[test]
fn canonical_fixture_metadata_pins_exact_order_shape_and_digest_index() {
    let fixture = CanonicalReplayFixture::load().unwrap();

    assert_eq!(fixture.manifest.name, "recorded-agent-eight-v1");
    assert_eq!(fixture.manifest.tasks.len(), 8);
    assert_eq!(
        fixture
            .manifest
            .tasks
            .iter()
            .map(|task| (&task.identity.adapter, &task.identity.task_id))
            .collect::<Vec<_>>(),
        vec![
            (
                &"pinchbench".to_string(),
                &"task_meeting_council_budget".to_string()
            ),
            (
                &"pinchbench".to_string(),
                &"task_meeting_council_votes".to_string()
            ),
            (&"pinchbench".to_string(), &"task_k8s_debugging".to_string()),
            (
                &"pinchbench".to_string(),
                &"task_meeting_searchable_index".to_string()
            ),
            (&"pinchbench".to_string(), &"task_skill_search".to_string()),
            (&"swebench".to_string(), &"django__django-15851".to_string()),
            (&"swebench".to_string(), &"django__django-14500".to_string()),
            (
                &"swebench".to_string(),
                &"sphinx-doc__sphinx-10614".to_string()
            ),
        ]
    );
    let expected = &fixture.manifest.aggregate;
    assert_eq!(expected.total_isl, 2_499_441);
    assert_eq!(expected.isl_delta, 192_314);
    assert_eq!(expected.peak_isl, 56_000);
    assert_eq!(expected.total_osl, 30_883);
    assert_eq!(expected.model_calls, 168);
    assert_eq!(expected.tool_calls, 172);
    assert!((expected.tool_duration_ms - 35_923.595_89).abs() <= 1e-6);
    assert!((expected.max_tool_call_duration_ms - 4_312.283731).abs() <= 1e-6);
    assert_eq!(expected.timed_out_tool_calls, 0);
    assert_eq!(fixture.manifest_digest, fixture.digest_index.manifest);
    assert_eq!(fixture.digest_index.recordings.len(), 8);
}

#[test]
#[ignore = "set AIPERF_RECORDED_AGENT_REPLAY_ROOT and run cargo test -p aiperf-runtime agent_recording --lib -- --ignored"]
fn canonical_fixture_validates_explicit_external_replay_root() {
    let root = std::env::var_os("AIPERF_RECORDED_AGENT_REPLAY_ROOT")
        .expect("ignored test requires AIPERF_RECORDED_AGENT_REPLAY_ROOT");
    let fixture = CanonicalReplayFixture::load().unwrap();
    let corpus = fixture.validate_replay_root(Path::new(&root)).unwrap();

    assert_eq!(corpus.traces.len(), 8);
    assert_eq!(corpus.recording_digests, fixture.digest_index.recordings);
}

fn defaults() -> serde_json::Value {
    json!({
        "config": "mixed", "step_limit": 1, "cost_limit": 0.0,
        "environment_class": "mixed", "docker_network": "none", "per_inference_timeout": 1.0,
        "fallback_max_output_tokens": 1, "temperature": 0.0, "top_p": 0.0, "top_k": 1,
        "min_p": 0.0, "stream_for_timing": true, "raw_openai_stream_for_replay_timing": true,
        "replay_max_tokens_from_recording": true, "replay_max_tokens_margin": 0,
        "extra_request_body": {}, "cross_run_cache_isolation": true, "warmup": true,
        "measurement_scope": "agent_run_only"
    })
}

fn shape() -> serde_json::Value {
    json!({
        "total_isl": 8, "isl_delta": 5, "peak_isl": 5, "total_osl": 6,
        "model_calls": 2, "tool_calls": 1, "tool_duration_ms": 4.0,
        "max_tool_call_duration_ms": 4.0, "timed_out_tool_calls": 0
    })
}

fn task(task_id: &str, recording: &str) -> serde_json::Value {
    json!({"adapter": "swebench", "family": "test", "task_id": task_id, "recording": recording})
}
