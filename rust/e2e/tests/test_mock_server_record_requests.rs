// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
mod common;
use common::*;

#[tokio::test]
#[ignore] // requires: Python aiperf_mock_server record-requests recorder (builtin tokenizer, inline JSONL + summary)
async fn test_records_per_request_isl_and_requested_osl() {}

#[tokio::test]
#[ignore] // requires: Python aiperf_mock_server.config.MockServerConfig record_requests validator
async fn test_record_requests_forces_workers_to_one() {}

#[tokio::test]
#[ignore] // requires: Python aiperf_mock_server.config.MockServerConfig record_requests/no_tokenizer validation
async fn test_record_requests_requires_a_tokenizer() {}
