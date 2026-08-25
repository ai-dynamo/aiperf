// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Full-stack regression coverage for synthesized Mooncake traces.

mod common;
use common::*;

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde_json::{Value, json};

const BLOCK_SIZE: u64 = 64;
const FULL_SHARED_PREFIX_TOKENS: u64 = 1_536;
const SESSION_COUNT: usize = 8;
const BASIC_BLOCK_SIZE: u64 = 512;
const BASIC_SESSION_COUNT: usize = 5;

/// Write a modest, block-aligned configuration for the generic synthesis
/// replay contract.
fn write_basic_config(dir: &Path) -> PathBuf {
    let config = json!({
        "block_size": BASIC_BLOCK_SIZE,
        "max_prompt_tokens": 10_000,
        "reset": null,
        "cache": {
            "layer1_tokens": 1_024,
            "layer1_5_tokens": 512,
            "layer2": {"mean": 400, "median": 300},
            "layer1_5_groups": {"num_groups": 3, "zipf_alpha": 1.2}
        }
    });
    let path = dir.join("basic.json");
    fs::write(
        &path,
        serde_json::to_vec_pretty(&config).expect("synthesis config serializes"),
    )
    .expect("write synthesis config");
    path
}

/// Write the upstream small-tail configuration at the shared-prefix boundary.
fn write_partial_prefix_config(dir: &Path) -> PathBuf {
    let config = json!({
        "block_size": BLOCK_SIZE,
        "max_prompt_tokens": 6_000,
        "reset": null,
        "cache": {
            "layer1_tokens": 1_000,
            "layer1_5_tokens": 500,
            "layer2": {"mean": 40, "median": 30},
            "layer1_5_groups": {"num_groups": 3, "zipf_alpha": 1.2}
        }
    });
    let path = dir.join("partial-prefix.json");
    fs::write(
        &path,
        serde_json::to_vec_pretty(&config).expect("synthesis config serializes"),
    )
    .expect("write synthesis config");
    path
}

fn synthesized_dataset(root: &Path) -> PathBuf {
    let mut directories = fs::read_dir(root)
        .expect("read synthesis output")
        .map(|entry| entry.expect("read synthesis output entry").path())
        .filter(|path| path.is_dir())
        .collect::<Vec<_>>();
    assert_eq!(
        directories.len(),
        1,
        "unexpected synthesis output: {directories:?}"
    );
    directories
        .pop()
        .expect("synthesized run directory")
        .join("dataset.jsonl")
}

fn read_trace(path: &Path) -> Vec<Value> {
    fs::read_to_string(path)
        .expect("read synthesized Mooncake trace")
        .lines()
        .map(|line| serde_json::from_str(line).expect("synthesized trace row is JSON"))
        .collect()
}

fn session_turns(rows: &[Value], id_path: &[&str]) -> BTreeMap<String, usize> {
    let mut turns = BTreeMap::new();
    for row in rows {
        let id = id_path
            .iter()
            .fold(row, |value, key| &value[*key])
            .as_str()
            .unwrap_or_else(|| panic!("row lacks session id at {id_path:?}: {row}"));
        *turns.entry(id.to_string()).or_default() += 1;
    }
    turns
}

fn synthesize_trace(
    h: &AIPerfHarness,
    config: &Path,
    output_name: &str,
    session_count: usize,
) -> (PathBuf, Vec<Value>, BTreeMap<String, usize>) {
    let synthesis_root = h.artifact_path().join(output_name);
    fs::create_dir(&synthesis_root).expect("create synthesis output directory");
    let synthesis = h.run_no_server(&format!(
        "synthesize agentic-code --num-sessions {session_count} --seed 42 --config {} --output {}",
        config.display(),
        synthesis_root.display(),
    ));
    assert!(
        synthesis.success(),
        "native synthesize failed:\n{}",
        synthesis.stderr
    );

    let trace_path = synthesized_dataset(&synthesis_root);
    let trace = read_trace(&trace_path);
    let trace_sessions = session_turns(&trace, &["session_id"]);
    assert_eq!(trace_sessions.len(), session_count);
    (trace_path, trace, trace_sessions)
}

fn assert_profile_replays_trace(
    h: &AIPerfHarness,
    trace_path: &Path,
    trace: &[Value],
    trace_sessions: &BTreeMap<String, usize>,
    block_size: u64,
    session_count: usize,
) {
    let replay = h.run(&format!(
        "--model {DEFAULT_MODEL} --url {} --endpoint-type chat \
         --input-file {} --custom-dataset-type mooncake_trace \
         --isl-block-size {block_size} --request-count {} \
         --concurrency {session_count} --workers-max 1 --ui simple",
        h.mock.url,
        trace_path.display(),
        trace.len(),
    ));
    assert!(
        replay.success(),
        "native profile failed:\n{}",
        replay.stderr
    );
    assert_eq!(replay.artifacts.request_count() as usize, trace.len());
    assert!(
        !replay.artifacts.json().is_null(),
        "missing summary artifact"
    );
    assert!(!replay.artifacts.csv().is_empty(), "missing CSV artifact");
    assert!(
        replay.artifacts.inputs().is_null(),
        "trace replay must not write inputs.json"
    );

    let records = replay.artifacts.jsonl();
    assert_eq!(records.len(), trace.len(), "missing profile records");
    let replay_sessions = session_turns(&records, &["metadata", "conversation_id"]);
    assert_eq!(
        replay_sessions.len(),
        session_count,
        "missing replay sessions"
    );
    assert_eq!(
        replay_sessions.keys().collect::<Vec<_>>(),
        trace_sessions.keys().collect::<Vec<_>>(),
        "replay session identities differ from the synthesized trace"
    );
    for record in records {
        assert!(
            record["metrics"]["input_sequence_length"]["value"]
                .as_f64()
                .is_some_and(|n| n > 0.0),
            "profile record has no input sequence length: {record}"
        );
        assert!(
            record["metrics"]["output_sequence_length"]["value"]
                .as_f64()
                .is_some_and(|n| n > 0.0),
            "profile record has no completion sequence length: {record}"
        );
    }
}

#[tokio::test]
async fn synthesized_trace_replays_through_profile() {
    let h = AIPerfHarness::new().await;
    let config = write_basic_config(h.artifact_path());
    let (trace_path, trace, trace_sessions) =
        synthesize_trace(&h, &config, "basic-synthesis", BASIC_SESSION_COUNT);
    assert_profile_replays_trace(
        &h,
        &trace_path,
        &trace,
        &trace_sessions,
        BASIC_BLOCK_SIZE,
        BASIC_SESSION_COUNT,
    );
}

#[tokio::test]
async fn synthesized_partial_prefix_trace_replays_through_profile() {
    let h = AIPerfHarness::new().await;
    let config = write_partial_prefix_config(h.artifact_path());
    let (trace_path, trace, trace_sessions) =
        synthesize_trace(&h, &config, "partial-prefix-synthesis", SESSION_COUNT);
    assert_profile_replays_trace(
        &h,
        &trace_path,
        &trace,
        &trace_sessions,
        BLOCK_SIZE,
        SESSION_COUNT,
    );

    // Both shared layers round up independently: 1000 -> 16 blocks and
    // 500 -> 8 blocks. The first session-owned token must follow all 24 full
    // shared blocks, so it cannot make the last shared hash id partial.
    let mut seen_sessions = BTreeMap::new();
    for row in &trace {
        let session = row["session_id"]
            .as_str()
            .expect("synthesized trace has session_id");
        if seen_sessions.insert(session, ()).is_none() {
            let input_length = row["input_length"]
                .as_u64()
                .expect("synthesized first turn has input_length");
            let hash_ids = row["hash_ids"]
                .as_array()
                .expect("synthesized first turn has hash_ids");
            assert!(
                input_length > FULL_SHARED_PREFIX_TOKENS,
                "{session} starts inside the 24-block shared prefix: {input_length}"
            );
            assert!(
                hash_ids.len() > (FULL_SHARED_PREFIX_TOKENS / BLOCK_SIZE) as usize,
                "{session} has no session-owned block: {hash_ids:?}"
            );
        }
    }
}
