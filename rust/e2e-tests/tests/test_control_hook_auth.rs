// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end coverage for control-hook authentication on the wire.
//!
//! An endpoint-local control hook POSTs to the same origin as inference, so an
//! authenticated endpoint rejects it unless the POST carries the endpoint's own
//! credentials and custom headers. These tests read the exact bytes the mock
//! server received on each control route.

mod common;
use common::*;

use aiperf_mock_server::RequestCapture;

const API_KEY: &str = "sk-control-hook-e2e";
const GATEWAY_HEADER: &str = "x-gateway-route";
const GATEWAY_VALUE: &str = "control-parity";
const CONTROL_ROUTES: [&str; 3] = ["/reset_prefix_cache", "/start_profile", "/stop_profile"];

fn mock_config() -> MockServerConfig {
    let mut cfg = MockServerConfig::default();
    cfg.fast = true;
    cfg.no_tokenizer = true;
    // Capture spans health, inference, and control traffic; leave room so the
    // three control POSTs cannot be evicted by benchmark requests.
    cfg.request_capture_capacity = 256;
    cfg
}

fn control_captures(harness: &AIPerfHarness, route: &str) -> Vec<RequestCapture> {
    harness
        .mock
        .state
        .request_captures()
        .into_iter()
        .filter(|capture| capture.route == route)
        .collect()
}

fn header(capture: &RequestCapture, name: &str) -> Option<String> {
    capture
        .header(name)
        .map(|value| String::from_utf8_lossy(value).into_owned())
}

#[tokio::test]
async fn control_posts_carry_endpoint_bearer_auth_and_authored_headers() {
    let h = AIPerfHarness::new_with(mock_config()).await;

    let r = h.run(&format!(
        "--model {DEFAULT_MODEL} --url {} --endpoint-type chat \
         --request-count 4 --concurrency 2 --workers-max 1 \
         --api-key {API_KEY} --header {GATEWAY_HEADER}:{GATEWAY_VALUE} \
         --reset-kv-cache --server-profiler --ui simple",
        h.mock.url
    ));
    assert!(r.success(), "authenticated control-hook run failed: {}", r.stderr);

    for route in CONTROL_ROUTES {
        let captures = control_captures(&h, route);
        assert_eq!(
            captures.len(),
            1,
            "expected exactly one {route} POST, got {}",
            captures.len()
        );
        let capture = &captures[0];
        assert_eq!(capture.method, "POST", "{route} method");
        assert_eq!(
            header(capture, "authorization").as_deref(),
            Some(format!("Bearer {API_KEY}").as_str()),
            "{route} must carry the endpoint bearer credential"
        );
        assert_eq!(
            header(capture, GATEWAY_HEADER).as_deref(),
            Some(GATEWAY_VALUE),
            "{route} must carry the authored endpoint header"
        );
    }
}

#[tokio::test]
async fn control_posts_send_no_authorization_without_a_credential() {
    let h = AIPerfHarness::new_with(mock_config()).await;

    let r = h.run(&format!(
        "--model {DEFAULT_MODEL} --url {} --endpoint-type chat \
         --request-count 4 --concurrency 2 --workers-max 1 \
         --reset-kv-cache --server-profiler --ui simple",
        h.mock.url
    ));
    assert!(r.success(), "credential-free control-hook run failed: {}", r.stderr);

    for route in CONTROL_ROUTES {
        let captures = control_captures(&h, route);
        assert_eq!(captures.len(), 1, "expected exactly one {route} POST");
        assert!(
            header(&captures[0], "authorization").is_none(),
            "{route} must not invent a credential: {:?}",
            captures[0].headers
        );
    }
}
