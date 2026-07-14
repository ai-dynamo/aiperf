// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
mod common;
use common::*;

use aiperf_mock_server::config::MockServerConfig;

// Integration tests for the `--wait-for-model-timeout` readiness probe.
//
// Covers both probe modes:
//
// Models mode (`--wait-for-model-mode models`):
// - success immediately (models endpoint ready from t=0)
// - success after N retries (models endpoint returns empty data until delay elapses)
// - timeout failure (requested model never appears)
// - 404 fallback (models endpoint disabled; probe accepts 2xx on base URL)
//
// Inference mode (`--wait-for-model-mode inference`, the default):
// - success immediately (inference endpoint ready from t=0)
// - success after N retries (inference endpoint returns 503 until delay elapses)

/// Build a fast, single-worker mock config that advertises `mock-model` on
/// `GET /v1/models`.
fn mock_model_cfg() -> MockServerConfig {
    let mut cfg = MockServerConfig::default();
    cfg.fast = true;
    cfg.workers = 1;
    cfg.models = vec!["mock-model".to_string()];
    cfg
}

// ---------------------------------------------------------------------------
// Models mode: `--wait-for-model-mode models`
// ---------------------------------------------------------------------------

/// With no configured delay, /v1/models lists the model from the start and the
/// probe returns on the first attempt.
#[tokio::test]
async fn test_models_probe_success_immediate() {
    let h = AIPerfHarness::new_with(mock_model_cfg()).await;
    let r = h.run_timeout(
        &format!(
            "--model mock-model --url {} --endpoint-type chat --streaming \
             --concurrency 1 --request-count 1 --workers-max 1 --ui simple \
             --wait-for-model-mode models --wait-for-model-timeout 30 \
             --wait-for-model-interval 1",
            h.mock.url
        ),
        120,
    );
    assert_eq!(r.exit_code, 0);
    let combined = format!("{}\n{}", r.stdout, r.stderr);
    assert!(
        combined.contains("Model 'mock-model' ready"),
        "expected readiness log, got:\n{combined}"
    );
}

/// With models_ready_delay_seconds>0, the probe sees an empty data list on
/// early attempts and must retry until the model appears.
///
/// requires: mock server `models_ready_delay_seconds` support (not implemented
/// in aiperf-mock-rs).
#[tokio::test]
#[ignore]
async fn test_models_probe_success_after_retries() {
    let mut cfg = mock_model_cfg();
    // models_ready_delay_seconds is unsupported by the Rust mock; this would set
    // it to 20.0 to force the empty-data retry path.
    cfg.workers = 1;
    let h = AIPerfHarness::new_with(cfg).await;
    let r = h.run_timeout(
        &format!(
            "--model mock-model --url {} --endpoint-type chat --streaming \
             --concurrency 1 --request-count 1 --workers-max 1 --ui simple \
             --wait-for-model-mode models --wait-for-model-timeout 60 \
             --wait-for-model-interval 0.5",
            h.mock.url
        ),
        180,
    );
    assert_eq!(r.exit_code, 0);
    let combined = format!("{}\n{}", r.stdout, r.stderr);
    assert!(combined.contains("not yet in"), "got:\n{combined}");
    assert!(
        combined.contains("Model 'mock-model' ready"),
        "got:\n{combined}"
    );
}

/// If the requested model id never appears in /v1/models, the probe must exit
/// non-zero and the error must reference the model and URL.
#[tokio::test]
async fn test_models_probe_timeout() {
    let h = AIPerfHarness::new_with(mock_model_cfg()).await;
    let r = h.run_timeout(
        &format!(
            "--model this-model-is-never-served --url {} --endpoint-type chat --streaming \
             --concurrency 1 --request-count 1 --workers-max 1 --ui simple \
             --wait-for-model-mode models --wait-for-model-timeout 3 \
             --wait-for-model-interval 0.5",
            h.mock.url
        ),
        60,
    );
    assert_ne!(r.exit_code, 0);
    let combined = format!("{}\n{}", r.stdout, r.stderr);
    assert!(
        combined.contains("this-model-is-never-served"),
        "got:\n{combined}"
    );
    assert!(combined.contains(&h.mock.url), "got:\n{combined}");
    assert!(combined.contains("Timed out"), "got:\n{combined}");
}

/// When /v1/models returns 404, the probe must fall back to a base-URL GET and
/// accept a 2xx as 'server is up'.
///
/// requires: mock server `disable_models_endpoint` support (not implemented in
/// aiperf-mock-rs).
#[tokio::test]
#[ignore]
async fn test_models_probe_404_fallback() {
    let h = AIPerfHarness::new_with(mock_model_cfg()).await;
    let r = h.run_timeout(
        &format!(
            "--model mock-model --url {} --endpoint-type chat --streaming \
             --concurrency 1 --request-count 1 --workers-max 1 --ui simple \
             --wait-for-model-mode models --wait-for-model-timeout 15 \
             --wait-for-model-interval 1",
            h.mock.url
        ),
        120,
    );
    assert_eq!(r.exit_code, 0);
    let combined = format!("{}\n{}", r.stdout, r.stderr);
    assert!(combined.contains("accepting as ready"), "got:\n{combined}");
}

// ---------------------------------------------------------------------------
// Inference mode: `--wait-for-model-mode inference`
// ---------------------------------------------------------------------------

/// With no configured delay, /v1/chat/completions responds 200 from t=0 and the
/// probe returns on the first attempt.
#[tokio::test]
async fn test_inference_probe_success_immediate() {
    let mut cfg = MockServerConfig::default();
    cfg.fast = true;
    cfg.workers = 1;
    let h = AIPerfHarness::new_with(cfg).await;
    let r = h.run_timeout(
        &format!(
            "--model mock-model --url {} --endpoint-type chat --streaming \
             --concurrency 1 --request-count 1 --workers-max 1 --ui simple \
             --wait-for-model-mode inference --wait-for-model-timeout 30 \
             --wait-for-model-interval 1",
            h.mock.url
        ),
        120,
    );
    assert_eq!(r.exit_code, 0);
    let combined = format!("{}\n{}", r.stdout, r.stderr);
    assert!(
        combined.contains("Inference probe ready"),
        "got:\n{combined}"
    );
}

/// With inference_ready_delay_seconds>0, the inference endpoint returns 503 on
/// early attempts and the probe must retry until the stack starts responding
/// 2xx.
///
/// requires: mock server `inference_ready_delay_seconds` support (not
/// implemented in aiperf-mock-rs).
#[tokio::test]
#[ignore]
async fn test_inference_probe_success_after_retries() {
    let mut cfg = MockServerConfig::default();
    cfg.fast = true;
    cfg.workers = 1;
    let h = AIPerfHarness::new_with(cfg).await;
    let r = h.run_timeout(
        &format!(
            "--model mock-model --url {} --endpoint-type chat --streaming \
             --concurrency 1 --request-count 1 --workers-max 1 --ui simple \
             --wait-for-model-mode inference --wait-for-model-timeout 60 \
             --wait-for-model-interval 0.5",
            h.mock.url
        ),
        180,
    );
    assert_eq!(r.exit_code, 0);
    let combined = format!("{}\n{}", r.stdout, r.stderr);
    assert!(combined.contains("returned 503"), "got:\n{combined}");
    assert!(
        combined.contains("Inference probe ready"),
        "got:\n{combined}"
    );
}
