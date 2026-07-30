// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Dry-run ports of component integration coverage outside the timing package.

mod common;

use std::collections::BTreeMap;

use common::{assert_credits_balanced, assert_request_count, run};

#[test]
fn sequence_distribution_generates_authored_length_buckets() {
    let run = run(&[
        "--num-sessions",
        "20",
        "--sequence-distribution",
        "128|0,64|0:100",
        "--random-seed",
        "42",
    ]);
    run.assert_success();
    assert_request_count(&run, 20, "sequence distribution").expect("all requests");
    for record in run.artifacts.jsonl() {
        assert_eq!(
            common::Artifacts::metric(&record, "input_sequence_length"),
            128.0
        );
        assert_eq!(
            common::Artifacts::metric(&record, "output_sequence_length"),
            64.0
        );
    }
}

#[test]
fn raw_payload_dataset_replays_authored_requests_through_dry_run() {
    let files = tempfile::tempdir().expect("payload directory");
    let path = files.path().join("payloads.jsonl");
    std::fs::write(
        &path,
        concat!(
            "{\"model\":\"openai/gpt-oss-120b\",\"messages\":[{\"role\":\"user\",\"content\":\"first\"}],\"max_tokens\":7}\n",
            "{\"model\":\"openai/gpt-oss-120b\",\"messages\":[{\"role\":\"user\",\"content\":\"second\"}],\"max_tokens\":9}\n",
        ),
    )
    .expect("write raw payloads");
    let path = path.to_str().expect("UTF-8 payload path");
    let run = run(&[
        "--custom-dataset-type",
        "raw_payload",
        "--input-file",
        path,
        "--num-conversations",
        "2",
        "--concurrency",
        "1",
    ]);
    run.assert_success();
    assert_request_count(&run, 2, "raw payload replay").expect("both payloads");
    assert_credits_balanced(&run).expect("raw payload credits");
}

#[test]
fn cancellation_records_http_499_and_releases_each_credit() {
    let run = run(&[
        "--num-sessions",
        "20",
        "--concurrency",
        "5",
        "--request-cancellation-rate",
        "100",
        "--request-cancellation-delay",
        "0",
        "--random-seed",
        "42",
    ]);
    run.assert_success();
    assert_request_count(&run, 20, "cancelled requests").expect("all requests terminal");
    assert_credits_balanced(&run).expect("cancelled request credits");
    for record in run.artifacts.jsonl() {
        assert_eq!(record["error"]["code"], 499, "record: {record}");
        assert_eq!(
            record["error"]["type"], "RequestCancellationError",
            "record: {record}"
        );
        assert_eq!(
            record["metadata"]["was_cancelled"], true,
            "record: {record}"
        );
        assert!(
            record["metadata"]["cancellation_time_ns"].is_number(),
            "record: {record}"
        );
    }
}

#[test]
fn cancellation_continues_multi_turn_sessions() {
    let run = run(&[
        "--num-sessions",
        "8",
        "--session-turns-mean",
        "3",
        "--session-turns-stddev",
        "0",
        "--concurrency",
        "4",
        "--request-cancellation-rate",
        "100",
        "--request-cancellation-delay",
        "0",
        "--random-seed",
        "42",
    ]);
    run.assert_success();
    assert_request_count(&run, 24, "cancelled multi-turn requests").expect("every turn terminal");
    assert_credits_balanced(&run).expect("cancelled multi-turn credits");
    let mut turns_by_session = BTreeMap::new();
    for record in run.artifacts.jsonl() {
        assert_eq!(record["error"]["code"], 499, "record: {record}");
        let session = record["metadata"]["conversation_id"]
            .as_str()
            .expect("conversation id")
            .to_string();
        let turn = record["metadata"]["turn_index"]
            .as_u64()
            .expect("turn index");
        turns_by_session
            .entry(session)
            .or_insert_with(Vec::new)
            .push(turn);
    }
    for turns in turns_by_session.values_mut() {
        turns.sort_unstable();
        assert_eq!(turns, &[0, 1, 2]);
    }
}
