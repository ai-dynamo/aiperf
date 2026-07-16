// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
mod common;
use common::*;

use aiperf_mock_server::grpc_riva::{
    RIVA_ASR_TRANSCRIPT, RIVA_NATURAL_QUERY_ANSWER, RIVA_SENTIMENT_CLASS,
};

// Full-stack e2e for the mock server's NVIDIA Riva ASR/TTS/NLP gRPC services.
//
// Runs the real `python -m aiperf profile` CLI (native runner + its production
// gRPC Riva client `aiperf_runtime::endpoints::riva`) against `aiperf-mock-server`'s own
// `serve_grpc` listener, selected via `transport.type: grpc` + a `grpc://` URL +
// a `riva_*` endpoint. Proves the whole product path — Python frontend → runner
// gRPC Riva client → mock gRPC Riva server — works for unary, server-streaming,
// and bidirectional-streaming RPCs, and verifies the raw per-record JSONL
// (`--export-level raw`) carries the exact transcript / audio / NLP result the
// mock returned.
//
// Determinism: the mock returns fixed content (a canned transcript, a fixed PCM
// ramp, canned classification/answer results), so each raw record's decoded
// gRPC response — surfaced in the raw record as `responses[].text` (the codec's
// canonical JSON) — is exactly predictable and asserted against the mock's
// public content constants.

const CONCURRENCY: u32 = 2;
const REQUEST_COUNT: u32 = 6;

/// A Config-v2 YAML selecting a native gRPC Riva endpoint against `grpc_url`.
///
/// `audio` is included only when the endpoint consumes audio (ASR): the synthetic
/// composer attaches one audio clip per turn, which the runner's ASR endpoint
/// lowers into the `Recognize`/`StreamingRecognize` request. TTS and NLP consume
/// the synthetic text prompt instead. The harness appends `--artifact-dir` and
/// `--tokenizer`, which override the corresponding config fields.
fn riva_config(grpc_url: &str, endpoint_type: &str, streaming: bool, with_audio: bool) -> String {
    let audio_block = if with_audio {
        "\x20   audio:\n\
        \x20     batchSize: 1\n\
        \x20     length: {mean: 1.0, stddev: 0.0}\n\
        \x20     format: wav\n\
        \x20     sampleRates: [16.0]\n\
        \x20     depths: [16]\n\
        \x20     channels: 1\n"
    } else {
        ""
    };
    format!(
        "schemaVersion: \"2.0\"\n\
         benchmark:\n\
        \x20 models: [{DEFAULT_MODEL}]\n\
        \x20 endpoint:\n\
        \x20   urls: [\"{grpc_url}\"]\n\
        \x20   type: {endpoint_type}\n\
        \x20   streaming: {streaming}\n\
        \x20   waitForModelTimeout: 0.0\n\
        \x20 dataset:\n\
        \x20   type: synthetic\n\
        \x20   entries: {REQUEST_COUNT}\n\
        \x20   prompts:\n\
        \x20     isl: 16\n\
        \x20     osl: 8\n\
        {audio_block}\
        \x20 phases:\n\
        \x20   - name: profiling\n\
        \x20     type: concurrency\n\
        \x20     requests: {REQUEST_COUNT}\n\
        \x20     concurrency: {CONCURRENCY}\n\
        \x20 gpuTelemetry: {{enabled: false}}\n\
        \x20 serverMetrics: {{enabled: false}}\n\
        \x20 transport:\n\
        \x20   type: grpc\n\
        \x20 runtime:\n\
        \x20   ui: none\n"
    )
}

/// Start a gRPC-enabled harness and run one Riva config against it. Returns the
/// harness (kept alive by the caller so its artifact `TempDir` survives the
/// assertions) and the run result. Panics with the captured stdout/stderr if the
/// run fails.
async fn run_riva(
    endpoint_type: &str,
    streaming: bool,
    with_audio: bool,
) -> (AIPerfHarness, RunResult) {
    let h = AIPerfHarness::new_with_grpc().await;
    let grpc_url = h
        .mock
        .grpc_url
        .clone()
        .expect("mock started with grpc listener");
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg_file = tmp.path().join("riva.yaml");
    std::fs::write(
        &cfg_file,
        riva_config(&grpc_url, endpoint_type, streaming, with_audio),
    )
    .unwrap();

    let r = h.run(&format!(
        "--config {} --export-level raw",
        cfg_file.display()
    ));
    assert!(
        r.success(),
        "riva {endpoint_type} run failed (exit {}):\nstdout:\n{}\nstderr:\n{}",
        r.exit_code,
        r.stdout,
        r.stderr
    );
    (h, r)
}

/// The concatenated `responses[].text` (each a decoded gRPC response as canonical
/// JSON) of the first raw record, plus the total raw-record count.
fn first_record_text(r: &RunResult) -> (usize, String) {
    let records = r.artifacts.raw_records();
    assert!(
        !records.is_empty(),
        "expected profile_export_raw.jsonl to contain records"
    );
    let responses = records[0]
        .get("responses")
        .and_then(|v| v.as_array())
        .expect("raw record has a responses array");
    assert!(
        !responses.is_empty(),
        "first raw record has at least one response"
    );
    let joined = responses
        .iter()
        .filter_map(|resp| resp.get("text").and_then(|t| t.as_str()))
        .collect::<Vec<_>>()
        .join("\n");
    (records.len(), joined)
}

/// Assert every raw record reports a 200 status (successful gRPC RPC).
fn assert_all_ok(r: &RunResult) {
    for (index, record) in r.artifacts.raw_records().iter().enumerate() {
        let status = record.get("status").and_then(|v| v.as_i64());
        assert_eq!(
            status,
            Some(200),
            "raw record {index} should report gRPC OK (200), got {status:?}"
        );
    }
}

/// ASR unary `Recognize`: the raw record carries the canned transcript.
#[tokio::test]
async fn test_riva_asr_recognize_unary() {
    let (_h, r) = run_riva("riva_asr", false, true).await;
    assert_eq!(r.artifacts.request_count() as u32, REQUEST_COUNT);
    assert_all_ok(&r);
    let (count, text) = first_record_text(&r);
    assert_eq!(count, REQUEST_COUNT as usize);
    assert!(
        text.contains(RIVA_ASR_TRANSCRIPT),
        "ASR raw record must carry the transcript {RIVA_ASR_TRANSCRIPT:?}; got:\n{text}"
    );
}

/// ASR bidirectional `StreamingRecognize`: the raw record carries interim and
/// final transcripts (two streamed responses), both the canned transcript.
#[tokio::test]
async fn test_riva_asr_streaming_recognize_bidi() {
    let (_h, r) = run_riva("riva_asr", true, true).await;
    assert_eq!(r.artifacts.request_count() as u32, REQUEST_COUNT);
    assert_all_ok(&r);
    // The bidi handler emits one interim + one final response per request.
    let responses = r.artifacts.raw_records()[0]
        .get("responses")
        .and_then(|v| v.as_array())
        .map(|a| a.len())
        .unwrap_or(0);
    assert!(
        responses >= 2,
        "streaming ASR should stream at least 2 responses (interim + final), got {responses}"
    );
    let (_count, text) = first_record_text(&r);
    assert!(
        text.contains(RIVA_ASR_TRANSCRIPT) && text.contains("\"is_final\":true"),
        "streaming ASR raw record must carry the transcript and a final result; got:\n{text}"
    );
}

/// TTS unary `Synthesize`: the raw record carries a non-empty base64 audio field.
#[tokio::test]
async fn test_riva_tts_synthesize_unary() {
    let (_h, r) = run_riva("riva_tts", false, false).await;
    assert_eq!(r.artifacts.request_count() as u32, REQUEST_COUNT);
    assert_all_ok(&r);
    let (_count, text) = first_record_text(&r);
    let response: serde_json::Value =
        serde_json::from_str(text.lines().next().unwrap()).expect("TTS response is JSON");
    let audio = response
        .get("audio")
        .and_then(|v| v.as_str())
        .expect("TTS response carries an audio field");
    assert!(
        !audio.is_empty(),
        "TTS raw record must carry non-empty base64 audio; got:\n{text}"
    );
}

/// TTS server-streaming `SynthesizeOnline`: the raw record carries multiple audio
/// chunks, each a non-empty audio field.
#[tokio::test]
async fn test_riva_tts_synthesize_online_streaming() {
    let (_h, r) = run_riva("riva_tts", true, false).await;
    assert_eq!(r.artifacts.request_count() as u32, REQUEST_COUNT);
    assert_all_ok(&r);
    let responses = r.artifacts.raw_records()[0]
        .get("responses")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    assert!(
        responses.len() >= 2,
        "streaming TTS should stream multiple audio chunks, got {}",
        responses.len()
    );
    for chunk in &responses {
        let text = chunk.get("text").and_then(|t| t.as_str()).unwrap_or("");
        let value: serde_json::Value = serde_json::from_str(text).expect("TTS chunk is JSON");
        let audio = value.get("audio").and_then(|v| v.as_str()).unwrap_or("");
        assert!(
            !audio.is_empty(),
            "each TTS chunk carries audio; got:\n{text}"
        );
    }
}

/// NLP unary `NaturalQuery`: the raw record carries the canned answer.
#[tokio::test]
async fn test_riva_natural_query_unary() {
    let (_h, r) = run_riva("riva_natural_query", false, false).await;
    assert_eq!(r.artifacts.request_count() as u32, REQUEST_COUNT);
    assert_all_ok(&r);
    let (_count, text) = first_record_text(&r);
    let response: serde_json::Value =
        serde_json::from_str(text.lines().next().unwrap()).expect("NaturalQuery response is JSON");
    let answer = response
        .get("results")
        .and_then(|v| v.as_array())
        .and_then(|a| a.first())
        .and_then(|first| first.get("answer"))
        .and_then(|a| a.as_str())
        .expect("NaturalQuery response carries results[0].answer");
    assert_eq!(
        answer, RIVA_NATURAL_QUERY_ANSWER,
        "NaturalQuery raw record must carry the canned answer"
    );
}

/// NLP unary `ClassifyText`: the raw record carries the canned sentiment label.
#[tokio::test]
async fn test_riva_text_classify_unary() {
    let (_h, r) = run_riva("riva_text_classify", false, false).await;
    assert_eq!(r.artifacts.request_count() as u32, REQUEST_COUNT);
    assert_all_ok(&r);
    let (_count, text) = first_record_text(&r);
    assert!(
        text.contains(RIVA_SENTIMENT_CLASS),
        "ClassifyText raw record must carry the sentiment class {RIVA_SENTIMENT_CLASS:?}; got:\n{text}"
    );
}
