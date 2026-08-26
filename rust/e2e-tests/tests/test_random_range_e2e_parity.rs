// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Full Python-vs-native process parity for random-range synthetic requests.

mod common;
use common::*;

use std::path::{Path, PathBuf};

use aiperf_mock_server::RequestCapture;
use aiperf_runtime::dataset::{HuggingFaceTokenizer, TextTokenizer};
use serde_json::Value;

const REQUESTS: usize = 8;

struct ParityCase {
    name: &'static str,
    seed: u64,
    ratio: &'static str,
    style: &'static str,
    isl: u32,
    osl: u32,
    tokenizer: TokenizerCase,
}

#[derive(Clone, Copy)]
enum TokenizerCase {
    ZeroSpecialTokens,
    TwoSpecialTokens,
}

impl TokenizerCase {
    fn cli_value(self) -> String {
        match self {
            Self::ZeroSpecialTokens => fixture_tokenizer("zero_special").display().to_string(),
            Self::TwoSpecialTokens => fixture_tokenizer("two_special").display().to_string(),
        }
    }

    fn load(self) -> Box<dyn TextTokenizer> {
        match self {
            Self::ZeroSpecialTokens => Box::new(
                HuggingFaceTokenizer::from_directory(fixture_tokenizer("zero_special")).unwrap(),
            ),
            Self::TwoSpecialTokens => Box::new(
                HuggingFaceTokenizer::from_directory(fixture_tokenizer("two_special")).unwrap(),
            ),
        }
    }
}

fn fixture_tokenizer(specials: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join(format!("random_range_tokenizer_{specials}"))
}

fn captured_workload(state: &aiperf_mock_server::AppState) -> Vec<RequestCapture> {
    state
        .request_captures()
        .into_iter()
        .filter(|capture| capture.route == "/v1/chat/completions")
        .collect()
}

fn prompt_text(body: &[u8]) -> String {
    let value: Value = serde_json::from_slice(body).unwrap();
    value["messages"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|message| match &message["content"] {
            Value::String(text) => Some(text.clone()),
            Value::Array(parts) => Some(
                parts
                    .iter()
                    .filter_map(|part| part["text"].as_str())
                    .collect::<Vec<_>>()
                    .join(""),
            ),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("")
}

async fn run_case(case: &ParityCase) {
    let mut mock_config = MockServerConfig::default();
    mock_config.fast = true;
    mock_config.no_tokenizer = true;
    mock_config.request_capture_capacity = REQUESTS * 2;
    let h = AIPerfHarness::new_with(mock_config).await;
    let config_path = h.artifact_path().join(format!("{}.yaml", case.name));
    std::fs::write(
        &config_path,
        format!(
            r#"schemaVersion: "2.0"
randomSeed: {}
benchmark:
  model: parity-model
  endpoint:
    type: chat
    url: {}
    streaming: false
  runtime:
    workers: 1
  dataset:
    type: synthetic
    entries: {REQUESTS}
    sampling: sequential
    prompts:
      isl: {}
      osl: {}
      corpus: random
      random_range_ratio: {}
      random_corpus_style: {}
  phases:
    type: concurrency
    requests: {REQUESTS}
    concurrency: 1
"#,
            case.seed, h.mock.url, case.isl, case.osl, case.ratio, case.style,
        ),
    )
    .unwrap();
    let args = format!(
        "--config '{}' --tokenizer '{}' --ui simple",
        config_path.display(),
        case.tokenizer.cli_value(),
    );

    let python = h.run_env(
        &args,
        &[
            ("AIPERF_RUNTIME_ENGINE", "python"),
            // The harness normally forces hub-offline mode. Python's tokenizer
            // wrapper routes *all* names through the hub-cache resolver in that
            // mode, including absolute local fixture paths. Empty values keep
            // the subprocess network-independent while allowing Transformers to
            // recognize and load this checked-in directory directly.
            ("HF_HUB_OFFLINE", ""),
            ("TRANSFORMERS_OFFLINE", ""),
        ],
    );
    assert!(
        python.success(),
        "{} Python profile failed:\nstdout:\n{}\nstderr:\n{}",
        case.name,
        python.stdout,
        python.stderr
    );
    let python_captures = captured_workload(&h.mock.state);
    assert_eq!(
        python_captures.len(),
        REQUESTS,
        "{} Python captures",
        case.name
    );
    h.mock.state.clear_request_captures();

    let native = h.run(&args);
    assert!(
        native.success(),
        "{} native profile failed:\nstdout:\n{}\nstderr:\n{}",
        case.name,
        native.stdout,
        native.stderr
    );
    let native_captures = captured_workload(&h.mock.state);
    assert_eq!(
        native_captures.len(),
        REQUESTS,
        "{} native captures",
        case.name
    );

    let tokenizer = case.tokenizer.load();
    for (index, (python, native)) in python_captures.iter().zip(&native_captures).enumerate() {
        assert_eq!(
            python.method, native.method,
            "{} request {index} method",
            case.name
        );
        assert_eq!(
            python.route, native.route,
            "{} request {index} route",
            case.name
        );
        assert_eq!(
            python.header("content-type"),
            native.header("content-type"),
            "{} request {index} content-type",
            case.name
        );
        assert_eq!(
            python.body,
            native.body,
            "{} request {index} outbound UTF-8 body; python prompts={:?}; native prompts={:?}",
            case.name,
            python_captures
                .iter()
                .map(|capture| prompt_text(&capture.body))
                .collect::<Vec<_>>(),
            native_captures
                .iter()
                .map(|capture| prompt_text(&capture.body))
                .collect::<Vec<_>>()
        );
        let python_tokens = tokenizer.encode(&prompt_text(&python.body)).unwrap();
        let native_tokens = tokenizer.encode(&prompt_text(&native.body)).unwrap();
        assert_eq!(
            python_tokens, native_tokens,
            "{} request {index} emitted prompt token IDs",
            case.name
        );
    }
}

#[tokio::test]
async fn python_and_native_profiles_emit_byte_exact_random_range_requests() {
    let cases = [
        ParityCase {
            name: "vllm-zero-seed-0",
            seed: 0,
            ratio: "0",
            style: "vllm",
            isl: 12,
            osl: 6,
            tokenizer: TokenizerCase::ZeroSpecialTokens,
        },
        ParityCase {
            name: "vllm-split-seed-42",
            seed: 42,
            ratio: r#"{"input":0.25,"output":0.5}"#,
            style: "vllm",
            isl: 12,
            osl: 6,
            tokenizer: TokenizerCase::ZeroSpecialTokens,
        },
        ParityCase {
            name: "vllm-near-boundary-two-special",
            seed: u64::MAX - 2,
            ratio: "0.899999",
            style: "vllm",
            isl: 12,
            osl: 5,
            tokenizer: TokenizerCase::TwoSpecialTokens,
        },
        ParityCase {
            name: "sglang-zero-seed-0",
            seed: 0,
            ratio: "0",
            style: "sglang",
            isl: 9,
            osl: 5,
            tokenizer: TokenizerCase::ZeroSpecialTokens,
        },
        ParityCase {
            name: "sglang-midpoint-two-special",
            seed: 42,
            ratio: "0.5",
            style: "sglang",
            isl: 12,
            osl: 6,
            tokenizer: TokenizerCase::TwoSpecialTokens,
        },
        ParityCase {
            name: "sglang-upper-boundary-wide-seed",
            seed: 4_294_967_300,
            ratio: "1",
            style: "sglang",
            isl: 11,
            osl: 7,
            tokenizer: TokenizerCase::ZeroSpecialTokens,
        },
    ];
    for case in &cases {
        run_case(case).await;
    }
}
