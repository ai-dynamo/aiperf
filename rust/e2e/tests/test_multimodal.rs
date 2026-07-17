// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
mod common;
use common::*;

use serde_json::Value;

const REQUEST_COUNT: u32 = 10;
const CONCURRENCY: u32 = 2;
const WORKERS_MAX: u32 = 1;

fn has_openai_media(payload: &Value, content_type: &str) -> bool {
    if let Some(messages) = payload.get("messages").and_then(|m| m.as_array()) {
        for message in messages {
            if let Some(content) = message.get("content").and_then(|c| c.as_array()) {
                for item in content {
                    if item.get(content_type).is_some() {
                        return true;
                    }
                }
            }
        }
    }
    false
}

fn has_top_level_media(payload: &Value, media_attr: &str) -> bool {
    payload
        .get(media_attr)
        .and_then(|m| m.as_array())
        .map(|a| !a.is_empty())
        .unwrap_or(false)
}

fn has_input_media(inputs: &Value, media_attr: &str) -> bool {
    let content_type = match media_attr {
        "images" => "image_url",
        "audios" => "input_audio",
        "videos" => "video_url",
        other => other,
    };

    let data = match inputs.get("data").and_then(|d| d.as_array()) {
        Some(d) if !d.is_empty() => d,
        _ => return false,
    };

    for session in data {
        let payloads = match session.get("payloads").and_then(|p| p.as_array()) {
            Some(p) => p,
            None => continue,
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

fn has_input_images(inputs: &Value) -> bool {
    has_input_media(inputs, "images")
}

fn has_input_audio(inputs: &Value) -> bool {
    has_input_media(inputs, "audios")
}

#[tokio::test]
async fn test_images() {
    let h = AIPerfHarness::new().await;
    let r = h.run(&format!(
        "--model {DEFAULT_MODEL} --url {} --endpoint-type chat \
         --request-count {REQUEST_COUNT} --concurrency {CONCURRENCY} \
         --image-width-mean 64 --image-height-mean 64 \
         --workers-max {WORKERS_MAX} --ui simple",
        h.mock.url
    ));
    assert_eq!(r.artifacts.request_count() as u32, REQUEST_COUNT);
    assert!(has_input_images(&r.artifacts.inputs()));
}

#[tokio::test]
async fn test_audio() {
    let h = AIPerfHarness::new().await;
    let r = h.run(&format!(
        "--model {DEFAULT_MODEL} --url {} --endpoint-type chat \
         --request-count {REQUEST_COUNT} --concurrency {CONCURRENCY} \
         --audio-length-mean 0.1 \
         --workers-max {WORKERS_MAX} --ui simple",
        h.mock.url
    ));
    assert_eq!(r.artifacts.request_count() as u32, REQUEST_COUNT);
    assert!(has_input_audio(&r.artifacts.inputs()));
}

#[tokio::test]
async fn test_images_and_audio() {
    let h = AIPerfHarness::new().await;
    let r = h.run(&format!(
        "--model {DEFAULT_MODEL} --url {} --endpoint-type chat \
         --request-count {REQUEST_COUNT} --concurrency {CONCURRENCY} \
         --image-width-mean 64 --image-height-mean 64 \
         --audio-length-mean 0.1 \
         --workers-max {WORKERS_MAX} --ui simple",
        h.mock.url
    ));
    assert_eq!(r.artifacts.request_count() as u32, REQUEST_COUNT);
    assert!(has_input_images(&r.artifacts.inputs()));
    assert!(has_input_audio(&r.artifacts.inputs()));
}
