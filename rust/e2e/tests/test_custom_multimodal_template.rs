// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
mod common;
use common::*;

// Integration test for custom multi-modal endpoint using template format.

/// Test custom multi-modal endpoint with images and audio using custom template.
///
/// Verifies that:
/// 1. The template endpoint type works with a custom Jinja2 template
/// 2. All expected output artifacts are created
#[tokio::test]
async fn test_custom_multimodal_with_images_and_audio() {
    let h = AIPerfHarness::new().await;

    // Custom jinja2 template that matches the custom endpoint format.
    let template = r#"{
    "modality_bundle": {
        "text_fragments": {{ texts|tojson }},
        "visual_assets": {
            "images": {{ images|tojson }}
        },
        "audio_streams": {{ audios|tojson }}
    },
    "inference_params": {
        "model_id": {{ model|tojson }},
        "sampling_config": {
            "max_tokens": {{ max_tokens|tojson }}
        }
    }
}"#;

    // Write template to a file to avoid shell escaping issues.
    let tmp = tempfile::TempDir::new().unwrap();
    let template_file = write_text(tmp.path(), "custom_template.json", template);

    let r = h.run(&format!(
        "--model {DEFAULT_MODEL} \
         --url {}/v1/custom-multimodal \
         --endpoint-type template \
         --extra-inputs payload_template:{} \
         --request-count {DEFAULT_REQUEST_COUNT} \
         --concurrency {DEFAULT_CONCURRENCY} \
         --synthetic-input-tokens-mean 50 \
         --image-width-mean 64 \
         --image-height-mean 64 \
         --audio-length-mean 0.1 \
         --workers-max 1 \
         --ui simple",
        h.mock.url,
        template_file.display(),
    ));

    // Verify the benchmark completed successfully.
    assert!(r.success(), "run failed: {}", r.stderr);
    assert_eq!(r.artifacts.request_count() as u32, DEFAULT_REQUEST_COUNT);

    // Verify all expected output artifacts are created (has_all_outputs).
    assert!(!r.artifacts.json().is_null(), "missing aiperf.json");
    assert!(
        !r.artifacts.jsonl().is_empty(),
        "missing profile_export.jsonl"
    );
    assert!(
        r.artifacts.find_file("*aiperf.csv").is_some(),
        "missing aiperf.csv"
    );
    assert!(!r.artifacts.inputs().is_null(), "missing inputs.json");
}
