// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
mod common;
use common::*;

fn has_openai_media(payload: &serde_json::Value, content_type: &str) -> bool {
    let Some(messages) = payload.get("messages").and_then(|m| m.as_array()) else {
        return false;
    };
    for message in messages {
        if let Some(content) = message.get("content").and_then(|c| c.as_array()) {
            for item in content {
                if item
                    .get(content_type)
                    .map(|v| !v.is_null())
                    .unwrap_or(false)
                {
                    return true;
                }
            }
        }
    }
    false
}

fn has_top_level_media(payload: &serde_json::Value, media_attr: &str) -> bool {
    payload
        .get(media_attr)
        .and_then(|v| v.as_array())
        .map(|a| !a.is_empty())
        .unwrap_or(false)
}

fn has_input_media(inputs: &serde_json::Value, media_attr: &str) -> bool {
    let content_type = match media_attr {
        "images" => "image_url",
        "audios" => "input_audio",
        "videos" => "video_url",
        other => other,
    };

    let Some(data) = inputs.get("data").and_then(|d| d.as_array()) else {
        return false;
    };

    for session in data {
        let Some(payloads) = session.get("payloads").and_then(|p| p.as_array()) else {
            continue;
        };
        for payload in payloads {
            if has_openai_media(payload, content_type) {
                return true;
            }
            if has_top_level_media(payload, media_attr) {
                return true;
            }
        }
    }
    false
}

fn has_input_images(inputs: &serde_json::Value) -> bool {
    has_input_media(inputs, "images")
}

fn has_input_audio(inputs: &serde_json::Value) -> bool {
    has_input_media(inputs, "audios")
}

#[tokio::test]
async fn test_image_formats_jpeg() {
    let h = AIPerfHarness::new().await;
    let r = h.run(&format!(
        "--model {DEFAULT_MODEL} \
         --url {} \
         --endpoint-type chat \
         --request-count 2 \
         --concurrency 2 \
         --image-width-mean 128 \
         --image-height-mean 128 \
         --image-format jpeg \
         --workers-max 1 \
         --ui simple",
        h.mock.url
    ));
    assert!(r.success(), "run failed: {}", r.stderr);
    assert_eq!(r.artifacts.request_count() as u32, 2);
    assert!(has_input_images(&r.artifacts.inputs()));
}

#[tokio::test]
async fn test_image_formats_png() {
    let h = AIPerfHarness::new().await;
    let r = h.run(&format!(
        "--model {DEFAULT_MODEL} \
         --url {} \
         --endpoint-type chat \
         --request-count 2 \
         --concurrency 2 \
         --image-width-mean 128 \
         --image-height-mean 128 \
         --image-format png \
         --workers-max 1 \
         --ui simple",
        h.mock.url
    ));
    assert!(r.success(), "run failed: {}", r.stderr);
    assert_eq!(r.artifacts.request_count() as u32, 2);
    assert!(has_input_images(&r.artifacts.inputs()));
}

#[tokio::test]
async fn test_audio_formats_mp3() {
    let h = AIPerfHarness::new().await;
    let r = h.run(&format!(
        "--model {DEFAULT_MODEL} \
         --url {} \
         --endpoint-type chat \
         --request-count 2 \
         --concurrency 2 \
         --audio-length-mean 0.1 \
         --audio-format mp3 \
         --workers-max 1 \
         --ui simple",
        h.mock.url
    ));
    assert!(r.success(), "run failed: {}", r.stderr);
    assert_eq!(r.artifacts.request_count() as u32, 2);
    assert!(has_input_audio(&r.artifacts.inputs()));
}

#[tokio::test]
async fn test_audio_formats_wav() {
    let h = AIPerfHarness::new().await;
    let r = h.run(&format!(
        "--model {DEFAULT_MODEL} \
         --url {} \
         --endpoint-type chat \
         --request-count 2 \
         --concurrency 2 \
         --audio-length-mean 0.1 \
         --audio-format wav \
         --workers-max 1 \
         --ui simple",
        h.mock.url
    ));
    assert!(r.success(), "run failed: {}", r.stderr);
    assert_eq!(r.artifacts.request_count() as u32, 2);
    assert!(has_input_audio(&r.artifacts.inputs()));
}
