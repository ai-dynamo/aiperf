// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native-binary integration coverage for file-backed random-pool batches.

mod common;

use common::{Artifacts, assert_credits_balanced, assert_request_count, run_config};

#[test]
fn yaml_text_batch_reaches_the_real_random_pool_loader() {
    let files = tempfile::tempdir().expect("random-pool fixture directory");
    let path = files.path().join("pool.jsonl");
    std::fs::write(
        &path,
        concat!(
            "{\"text\":\"alpha\"}\n",
            "{\"text\":\"beta\"}\n",
        ),
    )
    .expect("write random-pool fixture");
    let yaml = format!(
        r#"schemaVersion: "2.0"
randomSeed: 17
benchmark:
  model: openai/gpt-oss-120b
  tokenizer: {{name: cl100k_base}}
  transport:
    type: dry_run
    clock: sim
    ttft_ms: 10
    itl_ms: 2
  endpoint:
    type: chat
    url: http://127.0.0.1:8000
    streaming: true
  dataset:
    type: file
    format: random_pool
    path: {}
    numConversations: 1
    prompts: {{batchSize: 3}}
    images: {{batchSize: 0}}
  profiling:
    type: concurrency
    sessions: 1
    concurrency: 1
  artifacts:
    dir: $ARTIFACT_DIR
    raw: true
    records: [jsonl]
"#,
        path.display()
    );

    let run = run_config(&yaml);
    run.assert_success();
    assert_request_count(&run, 1, "random-pool conversation").expect("one request");
    assert_credits_balanced(&run).expect("random-pool credits balance");
    let records = run.artifacts.jsonl();
    assert_eq!(
        Artifacts::metric(&records[0], "input_sequence_length"),
        3.0,
        "three one-token text samples must reach composition"
    );
    let raw = run.artifacts.raw_jsonl();
    let payload = raw[0]["payload"].to_string();
    let sampled_items = payload.matches("alpha").count() + payload.matches("beta").count();
    assert_eq!(sampled_items, 3, "batched request payload: {payload}");
}
