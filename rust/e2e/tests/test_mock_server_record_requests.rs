// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
mod common;
use common::*;

// Integration tests for the mock server's per-request ISL/OSL recorder mode.
//
// The recorder lives in the Python `aiperf_mock_server` FastAPI lifespan: each
// request is tokenized inline with the configured tokenizer and written as one
// JSONL line, and a sibling `.summary.json` file is emitted on server shutdown.
//
// The Rust harness (`aiperf-mock-server` + `AIPerfHarness`) exposes no
// `record_requests` config knob, no builtin-tokenizer recorder, and no direct
// aiohttp-style request driver, so these are ported faithfully but ignored:
// they exercise the Python mock server package, not the Rust one.

/// Port of `test_records_per_request_isl_and_requested_osl`.
///
/// Drives five legacy `max_tokens` chat requests, one modern
/// `max_completion_tokens`/`min_tokens`/`reasoning_effort` chat request, one
/// completions request, and one embeddings request against a mock server
/// started with `record_requests=<path>` and `tokenizer=builtin`, then asserts
/// the per-request JSONL lines and the shutdown `.summary.json` block.
#[tokio::test]
#[ignore] // requires: Python aiperf_mock_server record-requests recorder (builtin tokenizer, inline JSONL + summary)
async fn test_records_per_request_isl_and_requested_osl() {
    // The Rust MockServerConfig/AIPerfHarness surface has no `record_requests`
    // option, no builtin-tokenizer inline recorder, and no direct HTTP POST
    // driver, so this recorder-summary behavior cannot be exercised here.
}

/// Port of `test_record_requests_forces_workers_to_one`.
///
/// The Python validator collapses `workers` to 1 whenever recording is on —
/// the recorder keeps per-request stats in-process, so a single uvicorn worker
/// is the supported producer. The Rust `MockServerConfig` has no
/// `record_requests` field and no such coupling.
#[tokio::test]
#[ignore] // requires: Python aiperf_mock_server.config.MockServerConfig record_requests validator
async fn test_record_requests_forces_workers_to_one() {
    // No equivalent `record_requests` -> workers=1 validator on the Rust config.
}

/// Port of `test_record_requests_requires_a_tokenizer`.
///
/// Recording counts ISL with the real tokenizer, so disabling the tokenizer
/// while requesting recording is incoherent and must fail fast with
/// "--record-requests requires a tokenizer". The Rust `MockServerConfig` has no
/// `record_requests`/`no_tokenizer` coupling to validate.
#[tokio::test]
#[ignore] // requires: Python aiperf_mock_server.config.MockServerConfig record_requests/no_tokenizer validation
async fn test_record_requests_requires_a_tokenizer() {
    // No equivalent record_requests + no_tokenizer ValidationError on the Rust config.
}
