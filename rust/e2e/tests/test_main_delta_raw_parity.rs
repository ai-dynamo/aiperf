// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Byte-exact `--export-level raw` e2e proofs for the origin/main delta port stages.
//!
//! Each stage runs `aiperf profile` against the in-process mock server twice (same
//! seed) and asserts deterministic raw/records projections are byte-identical
//! across the two runs, plus stage-specific observable contracts on the raw
//! artifact tree.

mod common;
use common::*;

use serde_json::{Value, json};
use std::collections::BTreeSet;
use std::path::Path;
use std::sync::{Mutex, MutexGuard, OnceLock};

fn timing_guard() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn sorted_strings(items: impl IntoIterator<Item = String>) -> Vec<String> {
    let mut v: Vec<String> = items.into_iter().collect();
    v.sort();
    v
}

/// Timing-free raw projection: identity + phase tags + request payload messages.
fn raw_projection(r: &Value) -> String {
    let m = &r["metadata"];
    json!({
        "session_num": m["session_num"],
        "conversation_id": m["conversation_id"],
        "turn_index": m["turn_index"],
        "benchmark_phase": m["benchmark_phase"],
        "phase_index": m["phase_index"],
        "phase_name": m["phase_name"],
        "phase_kind": m["phase_kind"],
        "profiling_index": m["profiling_index"],
        "status": r["status"],
        "payload": r["payload"],
    })
    .to_string()
}

/// Timing-free records.jsonl projection (dataset metrics + phase identity).
fn records_projection(r: &Value) -> String {
    let m = &r["metadata"];
    let met = &r["metrics"];
    json!({
        "session_num": m["session_num"],
        "conversation_id": m["conversation_id"],
        "turn_index": m["turn_index"],
        "benchmark_phase": m["benchmark_phase"],
        "phase_index": m["phase_index"],
        "phase_name": m["phase_name"],
        "phase_kind": m["phase_kind"],
        "profiling_index": m["profiling_index"],
        "input_sequence_length": met["input_sequence_length"],
        "output_sequence_length": met["output_sequence_length"],
        "error": r["error"],
    })
    .to_string()
}

fn assert_ab_raw_parity(a: &[Value], b: &[Value], label: &str) {
    assert_eq!(a.len(), b.len(), "{label}: raw record count diverged");
    assert_eq!(
        sorted_strings(a.iter().map(raw_projection)),
        sorted_strings(b.iter().map(raw_projection)),
        "{label}: raw deterministic projection SET diverged"
    );
}

fn assert_ab_records_parity(a: &[Value], b: &[Value], label: &str) {
    assert_eq!(a.len(), b.len(), "{label}: records.jsonl count diverged");
    assert_eq!(
        sorted_strings(a.iter().map(records_projection)),
        sorted_strings(b.iter().map(records_projection)),
        "{label}: records.jsonl deterministic projection SET diverged"
    );
}

fn write_config(dir: &Path, name: &str, body: &str) -> std::path::PathBuf {
    let path = dir.join(name);
    std::fs::write(&path, body).expect("write config");
    path
}

// ---------------------------------------------------------------------------
// Stage 0/1 — Mooncake `--isl-block-size` override + raw parity
// ---------------------------------------------------------------------------

#[tokio::test]
async fn stage01_mooncake_block_size_raw_parity() {
    let h = AIPerfHarness::new().await;
    // input_length 48 with 3 hash blocks is consistent with block_size 16 only.
    let traces = vec![
        json!({"input_length": 48, "output_length": 8, "hash_ids": [1, 2, 3], "timestamp": 100}),
        json!({"input_length": 48, "output_length": 8, "hash_ids": [4, 5, 6], "timestamp": 200}),
    ];
    let trace_file = write_jsonl(h.artifact_dir.path(), "mooncake.jsonl", &traces);
    let n = traces.len();
    let args = format!(
        "--model {DEFAULT_MODEL} --url {} --endpoint-type chat --streaming \
         --input-file {} --custom-dataset-type mooncake_trace \
         --isl-block-size 16 --request-count {n} --concurrency 1 \
         --workers-max 1 --random-seed 42 --export-level raw --ui simple \
         --tokenizer builtin",
        h.mock.url,
        trace_file.display()
    );

    let r1 = h.run(&args);
    assert!(r1.success(), "run1 failed: {}", r1.stderr);
    let raw1 = r1.artifacts.raw_records();
    let rec1 = r1.artifacts.jsonl();
    let inputs1 = r1.artifacts.inputs();

    let r2 = h.run(&args);
    assert!(r2.success(), "run2 failed: {}", r2.stderr);
    assert_ab_raw_parity(&raw1, &r2.artifacts.raw_records(), "mooncake block_size");
    assert_ab_records_parity(&rec1, &r2.artifacts.jsonl(), "mooncake block_size");
    assert_eq!(
        inputs1,
        r2.artifacts.inputs(),
        "mooncake block_size: inputs.json must be byte-identical across seeded runs"
    );

    assert_eq!(raw1.len(), n);
    assert_eq!(rec1.len(), n);
    for rec in &rec1 {
        let isl = rec["metrics"]["input_sequence_length"]["value"]
            .as_f64()
            .or_else(|| rec["metrics"]["input_sequence_length"].as_f64())
            .expect("ISL present");
        assert!(
            (isl - 48.0).abs() < 1e-9,
            "expected reconstructed ISL=48 with --isl-block-size 16, got {isl}"
        );
    }
}

// ---------------------------------------------------------------------------
// Stage 1 — NVIDIA telemetry namespace with raw export
// ---------------------------------------------------------------------------

#[tokio::test]
async fn stage1_nvidia_telemetry_raw_parity() {
    if cfg!(target_os = "windows") || cfg!(target_os = "macos") {
        return;
    }
    let h = AIPerfHarness::new().await;
    let dcgm = h.mock.dcgm_urls().join(" ");
    let args = format!(
        "--model {DEFAULT_MODEL} --url {} --endpoint-type chat --streaming \
         --gpu-telemetry {dcgm} --request-count 8 --concurrency 2 \
         --workers-max 1 --random-seed 7 --export-level raw --ui simple \
         --synthetic-input-tokens-mean 32 --synthetic-input-tokens-stddev 0 \
         --output-tokens-mean 8 --output-tokens-stddev 0 --tokenizer builtin",
        h.mock.url
    );

    let r1 = h.run(&args);
    assert!(r1.success(), "run1 failed: {}", r1.stderr);
    let raw1 = r1.artifacts.raw_records();
    let json1 = r1.artifacts.json();

    let r2 = h.run(&args);
    assert!(r2.success(), "run2 failed: {}", r2.stderr);
    assert_ab_raw_parity(&raw1, &r2.artifacts.raw_records(), "nvidia telemetry");

    let endpoints = json1
        .pointer("/telemetry_data/endpoints")
        .and_then(Value::as_object)
        .expect("telemetry_data.endpoints");
    assert!(!endpoints.is_empty(), "expected GPU telemetry endpoints");
    let mut saw_nvidia_power = false;
    for endpoint in endpoints.values() {
        let gpus = endpoint["gpus"].as_object().expect("gpus");
        for gpu in gpus.values() {
            let metrics = gpu["metrics"].as_object().expect("metrics");
            assert!(
                !metrics.contains_key("gpu_power_usage"),
                "legacy gpu_power_usage must not appear after nvidia_* rename"
            );
            if metrics.contains_key("nvidia_power_usage") {
                saw_nvidia_power = true;
            }
        }
    }
    assert!(
        saw_nvidia_power,
        "expected nvidia_power_usage in telemetry summary"
    );
    assert_eq!(raw1.len(), 8);
}

// ---------------------------------------------------------------------------
// Stage 2 — Rate series QPS curve with tuned mock + raw timestamps
// ---------------------------------------------------------------------------

#[tokio::test]
async fn stage2_rate_series_raw_parity_and_qps_curve() {
    if cfg!(target_os = "macos") {
        return;
    }
    let _guard = timing_guard();
    let h = AIPerfHarness::new_with(tuned_mock_config(20.0, 5.0)).await;

    let series_path = h.artifact_dir.path().join("rate_series.json");
    // Hold 5 QPS for 2s, then 10 QPS for 2s — constant arrival for knife-edge spacing.
    std::fs::write(
        &series_path,
        r#"[{"time_s":0,"qps":5},{"time_s":2,"qps":5},{"time_s":2.01,"qps":10},{"time_s":4,"qps":10}]"#,
    )
    .unwrap();

    let cfg = write_config(
        h.artifact_dir.path(),
        "rate_series.yaml",
        &format!(
            "schemaVersion: \"2.0\"\n\
             randomSeed: 11\n\
             \n\
             benchmark:\n\
            \x20 model: gpt-4\n\
            \x20 endpoint:\n\
            \x20   url: {}/v1/chat/completions\n\
            \x20   type: chat\n\
            \x20   streaming: true\n\
            \x20 dataset:\n\
            \x20   type: synthetic\n\
            \x20   entries: 200\n\
            \x20   random_seed: 11\n\
            \x20   prompts:\n\
            \x20     isl: 32\n\
            \x20     osl: 4\n\
            \x20 phases:\n\
            \x20   - name: profiling\n\
            \x20     kind: profiling\n\
            \x20     type: constant\n\
            \x20     requests: 30\n\
            \x20     concurrency: 1\n\
            \x20     rateSeries: {}\n\
            \x20 artifacts:\n\
            \x20   records: [jsonl]\n\
            \x20   raw: true\n\
             \n\
             runtime:\n\
            \x20 workers: 1\n",
            h.mock.url,
            series_path.display()
        ),
    );

    let args = format!(
        "--config {} --ui simple --tokenizer builtin --export-level raw --random-seed 11",
        cfg.display()
    );
    let r1 = h.run(&args);
    assert!(r1.success(), "run1 failed: {}", r1.stderr);
    let raw1 = r1.artifacts.raw_records();
    let rec1 = r1.artifacts.jsonl();

    let r2 = h.run(&args);
    assert!(r2.success(), "run2 failed: {}", r2.stderr);
    assert_ab_raw_parity(&raw1, &r2.artifacts.raw_records(), "rate series");
    assert_ab_records_parity(&rec1, &r2.artifacts.jsonl(), "rate series");

    assert!(
        raw1.len() >= 20,
        "rate-series run should issue many requests, got {}",
        raw1.len()
    );
    assert_eq!(
        raw1.len(),
        30,
        "request-bounded rate-series must issue exactly 30"
    );

    // Issue times from credit_issued_ns (phase-relative admit). Sort and measure
    // early-window vs late-window mean inter-arrival.
    let mut issued: Vec<i64> = raw1
        .iter()
        .filter_map(|r| r["metadata"]["credit_issued_ns"].as_i64())
        .collect();
    issued.sort_unstable();
    assert!(
        issued.len() >= 20,
        "need credit_issued_ns on raw records for QPS proof"
    );
    let t0 = issued[0];
    let early: Vec<i64> = issued
        .iter()
        .copied()
        .filter(|t| *t - t0 < 1_800_000_000)
        .collect();
    let late: Vec<i64> = issued
        .iter()
        .copied()
        .filter(|t| *t - t0 > 2_200_000_000)
        .collect();
    assert!(
        early.len() >= 4 && late.len() >= 4,
        "need samples in both rate windows: early={} late={}",
        early.len(),
        late.len()
    );
    let mean_gap = |times: &[i64]| -> f64 {
        let gaps: Vec<i64> = times.windows(2).map(|w| w[1] - w[0]).collect();
        gaps.iter().sum::<i64>() as f64 / gaps.len() as f64
    };
    let early_gap = mean_gap(&early);
    let late_gap = mean_gap(&late);
    // 5 QPS → ~200ms gap; 10 QPS → ~100ms gap. Allow 40% tolerance for loopback.
    assert!(
        early_gap > late_gap * 1.3,
        "early window (5 QPS) mean gap {early_gap} ns should be >> late window (10 QPS) gap {late_gap} ns"
    );
    assert!(
        (150_000_000.0..280_000_000.0).contains(&early_gap),
        "early mean gap {early_gap} ns outside ~200ms band for 5 QPS"
    );
    assert!(
        (60_000_000.0..160_000_000.0).contains(&late_gap),
        "late mean gap {late_gap} ns outside ~100ms band for 10 QPS"
    );
}

// ---------------------------------------------------------------------------
// Stage 3 — Named multi-phase identity + phase_manifest + raw parity
// ---------------------------------------------------------------------------

#[tokio::test]
async fn stage3_named_multiphase_raw_parity_and_manifest() {
    let h = AIPerfHarness::new().await;
    let cfg = write_config(
        h.artifact_dir.path(),
        "multiphase.yaml",
        &format!(
            "schemaVersion: \"2.0\"\n\
             randomSeed: 99\n\
             \n\
             benchmark:\n\
            \x20 model: {DEFAULT_MODEL}\n\
            \x20 endpoint:\n\
            \x20   url: {}/v1/chat/completions\n\
            \x20   type: chat\n\
            \x20   streaming: true\n\
            \x20 dataset:\n\
            \x20   type: synthetic\n\
            \x20   entries: 32\n\
            \x20   random_seed: 99\n\
            \x20   prompts:\n\
            \x20     isl: 24\n\
            \x20     osl: 8\n\
            \x20 phases:\n\
            \x20   - name: setup\n\
            \x20     kind: warmup\n\
            \x20     type: concurrency\n\
            \x20     concurrency: 1\n\
            \x20     requests: 2\n\
            \x20   - name: low\n\
            \x20     kind: profiling\n\
            \x20     type: concurrency\n\
            \x20     concurrency: 1\n\
            \x20     requests: 3\n\
            \x20   - name: storm\n\
            \x20     kind: profiling\n\
            \x20     type: concurrency\n\
            \x20     concurrency: 2\n\
            \x20     requests: 4\n\
            \x20 artifacts:\n\
            \x20   records: [jsonl]\n\
            \x20   raw: true\n\
             \n\
             runtime:\n\
            \x20 workers: 1\n",
            h.mock.url
        ),
    );

    let args = format!(
        "--config {} --ui simple --no-gpu-telemetry --tokenizer builtin \
         --export-level raw --random-seed 99",
        cfg.display()
    );
    let r1 = h.run(&args);
    assert!(r1.success(), "run1 failed: {}", r1.stderr);
    let raw1 = r1.artifacts.raw_records();
    let rec1 = r1.artifacts.jsonl();
    let manifest1 = std::fs::read_to_string(
        r1.artifacts
            .find_file("**/phase_manifest.json")
            .expect("phase_manifest.json"),
    )
    .unwrap();

    let r2 = h.run(&args);
    assert!(r2.success(), "run2 failed: {}", r2.stderr);
    assert_ab_raw_parity(&raw1, &r2.artifacts.raw_records(), "multiphase");
    assert_ab_records_parity(&rec1, &r2.artifacts.jsonl(), "multiphase");
    let manifest2 = std::fs::read_to_string(
        r2.artifacts
            .find_file("**/phase_manifest.json")
            .expect("phase_manifest.json"),
    )
    .unwrap();
    assert_eq!(
        manifest1, manifest2,
        "phase_manifest.json must be byte-identical across seeded runs"
    );

    let manifest: Value = serde_json::from_str(&manifest1).unwrap();
    assert_eq!(manifest["schema_version"], 1);
    let phases = manifest["phases"].as_array().expect("phases");
    assert_eq!(phases.len(), 3);
    assert_eq!(phases[0]["phase_name"], "setup");
    assert_eq!(phases[0]["phase_kind"], "warmup");
    assert!(phases[0]["profiling_index"].is_null());
    assert_eq!(phases[1]["phase_name"], "low");
    assert_eq!(phases[1]["profiling_index"], 0);
    assert_eq!(phases[2]["phase_name"], "storm");
    assert_eq!(phases[2]["profiling_index"], 1);

    // Profiling raw records only (warmup excluded from aggregate export counts
    // may still appear in raw depending on export policy — assert named tags).
    let mut names = BTreeSet::new();
    for rec in &raw1 {
        if let Some(name) = rec["metadata"]["phase_name"].as_str() {
            names.insert(name.to_string());
            let kind = rec["metadata"]["phase_kind"].as_str();
            match name {
                "setup" => {
                    assert_eq!(rec["metadata"]["phase_index"], 0);
                    assert_eq!(kind, Some("warmup"));
                    assert!(rec["metadata"]["profiling_index"].is_null());
                }
                "low" => {
                    assert_eq!(rec["metadata"]["phase_index"], 1);
                    assert_eq!(kind, Some("profiling"));
                    assert_eq!(rec["metadata"]["profiling_index"], 0);
                }
                "storm" => {
                    assert_eq!(rec["metadata"]["phase_index"], 2);
                    assert_eq!(kind, Some("profiling"));
                    assert_eq!(rec["metadata"]["profiling_index"], 1);
                }
                other => panic!("unexpected phase_name {other}"),
            }
        }
    }
    assert!(
        names.contains("low") && names.contains("storm"),
        "raw records must carry named profiling phases, got {names:?}"
    );
}

// ---------------------------------------------------------------------------
// Stage 4 — Reasoning channel split in raw + outputs parity
// ---------------------------------------------------------------------------

#[tokio::test]
async fn stage4_reasoning_content_raw_parity() {
    use aiperf_mock_server::accuracy::AccuracyFormat;
    use aiperf_mock_server::config::MockServerConfig;

    let ds_dir = tempfile::TempDir::new().unwrap();
    let records: Vec<Value> = (0..6)
        .map(|i| {
            json!({
                "text": format!("Question {i}: which option is correct, A, B, C, or D?"),
                "ground_truth": "B",
                "task": "demo",
            })
        })
        .collect();
    let dataset = write_jsonl(ds_dir.path(), "accuracy.jsonl", &records);

    let cfg = MockServerConfig {
        fast: true,
        no_tokenizer: true,
        random_seed: Some(7),
        accuracy_dataset: Some(dataset.to_string_lossy().into_owned()),
        accuracy_format: AccuracyFormat::Mmlu,
        accuracy_correct_rate: 1.0,
        accuracy_cot_rate: 1.0,
        accuracy_reasoning_field: true,
        ..MockServerConfig::default()
    };
    let h = AIPerfHarness::new_with(cfg).await;
    let args = format!(
        "--model gpt-4 --url {} --endpoint-type chat --streaming \
         --input-file {} --custom-dataset-type single_turn \
         --request-count 6 --concurrency 2 --workers-max 1 \
         --random-seed 7 --export-level raw --ui simple --tokenizer builtin",
        h.mock.url,
        dataset.display()
    );

    let r1 = h.run(&args);
    assert!(r1.success(), "run1 failed: {}", r1.stderr);
    let raw1 = r1.artifacts.raw_records();
    let outputs1 = r1
        .artifacts
        .find_file("**/outputs.json")
        .map(|p| std::fs::read_to_string(p).unwrap());

    let r2 = h.run(&args);
    assert!(r2.success(), "run2 failed: {}", r2.stderr);
    assert_ab_raw_parity(&raw1, &r2.artifacts.raw_records(), "reasoning accuracy");

    assert_eq!(raw1.len(), 6);
    for rec in &raw1 {
        let content = extract_delta_field(rec, "content");
        let reasoning = extract_delta_field(rec, "reasoning_content");
        assert_eq!(content, "The answer is (B)");
        assert!(
            !reasoning.is_empty(),
            "reasoning_content channel must be non-empty for CoT fixtures"
        );
        // Visible content must not be polluted with the CoT preamble.
        assert!(
            !content.contains("Thinking"),
            "visible content must exclude reasoning preamble, got {content:?}"
        );
    }

    if let Some(text) = outputs1 {
        let doc: Value = serde_json::from_str(&text).unwrap();
        let data = doc["data"].as_array().expect("outputs.data");
        for row in data {
            let response = row["response_text"].as_str().unwrap_or("");
            let thinking = row.get("reasoning_text").and_then(Value::as_str);
            assert!(
                !response.is_empty() || thinking.is_some(),
                "outputs row must carry visible and/or reasoning text: {row}"
            );
            if let Some(t) = thinking {
                assert!(!t.is_empty(), "reasoning_text present but empty");
            }
        }
    }
}

fn extract_delta_field(record: &Value, field: &str) -> String {
    let mut out = String::new();
    let Some(responses) = record.get("responses").and_then(Value::as_array) else {
        return out;
    };
    for resp in responses {
        let Some(packets) = resp.get("packets").and_then(Value::as_array) else {
            continue;
        };
        for packet in packets {
            if packet.get("name").and_then(Value::as_str) != Some("data") {
                continue;
            }
            let Some(raw) = packet.get("value").and_then(Value::as_str) else {
                continue;
            };
            let trimmed = raw.trim();
            if trimmed == "[DONE]" {
                continue;
            }
            if let Ok(obj) = serde_json::from_str::<Value>(trimmed)
                && let Some(c) = obj
                    .pointer(&format!("/choices/0/delta/{field}"))
                    .and_then(Value::as_str)
            {
                out.push_str(c);
            }
        }
    }
    out
}
