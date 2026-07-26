// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
mod common;
use common::*;

use aiperf_mock_server::config::MockServerConfig;

fn mock_model_cfg() -> MockServerConfig {
    let mut cfg = MockServerConfig::default();
    cfg.fast = true;
    cfg.workers = 1;
    cfg.models = vec!["mock-model".to_string()];
    cfg
}

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

#[tokio::test]
#[ignore = "requires mock-server control over model-list readiness delay"]
async fn test_models_probe_success_after_retries() {
    let mut cfg = mock_model_cfg();
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

#[tokio::test]
#[ignore = "requires mock-server control to disable the models endpoint"]
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

#[tokio::test]
#[ignore = "requires mock-server control over inference readiness delay"]
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
