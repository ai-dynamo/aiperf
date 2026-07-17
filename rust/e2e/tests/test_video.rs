// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
mod common;
use common::*;

use base64::Engine;
use std::io::Write;
use std::process::{Command, Stdio};

const WORKERS_MAX: u32 = 1;

struct VideoDetails {
    width: i64,
    height: i64,
    fps: f64,
    duration: f64,
    is_fragmented: bool,
    has_audio: bool,
    audio_codec: Option<String>,
    audio_channels: Option<i64>,
    audio_sample_rate: Option<i64>,
}

fn ffprobe_available() -> bool {
    Command::new("ffprobe")
        .arg("-version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn check_mp4_fragmentation(video_bytes: &[u8]) -> bool {
    let header_size = video_bytes.len().min(10240);
    video_bytes[..header_size].windows(4).any(|w| w == b"moof")
}

fn parse_rational_fps(s: &str) -> f64 {
    if let Some((num, den)) = s.split_once('/') {
        let n: f64 = num.parse().unwrap_or(0.0);
        let d: f64 = den.parse().unwrap_or(1.0);
        if d != 0.0 {
            return n / d;
        }
        return 0.0;
    }
    s.parse().unwrap_or(0.0)
}

fn extract_base64_video_details(base64_data: &str) -> VideoDetails {
    let video_bytes = base64::engine::general_purpose::STANDARD
        .decode(base64_data.trim())
        .expect("failed to base64-decode video data");

    let mut child = Command::new("ffprobe")
        .args([
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            "-",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("failed to spawn ffprobe");

    child
        .stdin
        .take()
        .unwrap()
        .write_all(&video_bytes)
        .expect("failed to write video bytes to ffprobe");

    let output = child.wait_with_output().expect("ffprobe failed");
    let probe: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("failed to parse ffprobe json");

    let empty = Vec::new();
    let streams = probe
        .get("streams")
        .and_then(|s| s.as_array())
        .unwrap_or(&empty);

    let video_stream = streams
        .iter()
        .find(|s| s.get("codec_type").and_then(|c| c.as_str()) == Some("video"));
    let audio_stream = streams
        .iter()
        .find(|s| s.get("codec_type").and_then(|c| c.as_str()) == Some("audio"));

    let width = video_stream
        .and_then(|s| s.get("width"))
        .and_then(|v| v.as_i64())
        .unwrap_or(0);
    let height = video_stream
        .and_then(|s| s.get("height"))
        .and_then(|v| v.as_i64())
        .unwrap_or(0);
    let fps = video_stream
        .and_then(|s| s.get("avg_frame_rate").or_else(|| s.get("r_frame_rate")))
        .and_then(|v| v.as_str())
        .map(parse_rational_fps)
        .unwrap_or(0.0);

    let duration = probe
        .get("format")
        .and_then(|f| f.get("duration"))
        .and_then(|d| d.as_str())
        .and_then(|d| d.parse::<f64>().ok())
        .or_else(|| {
            video_stream
                .and_then(|s| s.get("duration"))
                .and_then(|d| d.as_str())
                .and_then(|d| d.parse::<f64>().ok())
        })
        .unwrap_or(0.0);

    let has_audio = audio_stream.is_some();
    let audio_codec = audio_stream
        .and_then(|s| s.get("codec_name"))
        .and_then(|c| c.as_str())
        .map(|s| s.to_string());
    let audio_channels = audio_stream
        .and_then(|s| s.get("channels"))
        .and_then(|c| c.as_i64());
    let audio_sample_rate = audio_stream
        .and_then(|s| s.get("sample_rate"))
        .and_then(|c| c.as_str())
        .and_then(|c| c.parse::<i64>().ok());

    VideoDetails {
        width,
        height,
        fps,
        duration,
        is_fragmented: check_mp4_fragmentation(&video_bytes),
        has_audio,
        audio_codec,
        audio_channels,
        audio_sample_rate,
    }
}

fn iter_video_details(inputs: &serde_json::Value) -> Vec<VideoDetails> {
    let mut out = Vec::new();
    let Some(data) = inputs.get("data").and_then(|d| d.as_array()) else {
        return out;
    };
    for session in data {
        let Some(payloads) = session.get("payloads").and_then(|p| p.as_array()) else {
            continue;
        };
        for payload in payloads {
            let Some(messages) = payload.get("messages").and_then(|m| m.as_array()) else {
                continue;
            };
            for message in messages {
                let Some(content) = message.get("content").and_then(|c| c.as_array()) else {
                    continue;
                };
                for item in content {
                    let Some(video_url) = item.get("video_url").and_then(|v| v.as_object()) else {
                        continue;
                    };
                    let Some(url) = video_url.get("url").and_then(|u| u.as_str()) else {
                        continue;
                    };
                    if let Some((_, data)) = url.split_once(',') {
                        out.push(extract_base64_video_details(data));
                    }
                }
            }
        }
    }
    out
}

fn has_input_videos(inputs: &serde_json::Value) -> bool {
    let Some(data) = inputs.get("data").and_then(|d| d.as_array()) else {
        return false;
    };
    for session in data {
        let Some(payloads) = session.get("payloads").and_then(|p| p.as_array()) else {
            continue;
        };
        for payload in payloads {
            let Some(messages) = payload.get("messages").and_then(|m| m.as_array()) else {
                continue;
            };
            for message in messages {
                if let Some(content) = message.get("content").and_then(|c| c.as_array()) {
                    for item in content {
                        if item.get("video_url").map(|v| !v.is_null()).unwrap_or(false) {
                            return true;
                        }
                    }
                }
            }
        }
    }
    false
}

fn approx_eq(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-6 + 1e-6 * b.abs()
}

async fn video_generation_parameters(
    video_format: &str,
    video_codec: &str,
    check_fragmentation: bool,
) {
    if cfg!(target_os = "windows") || !ffprobe_available() {
        return;
    }
    let (width, height, fps, duration) = (512, 288, 4, 5.0);

    let h = AIPerfHarness::new().await;
    let r = h.run(&format!(
        "--model {DEFAULT_MODEL} --url {} --endpoint-type chat \
         --video-width {width} --video-height {height} --video-duration {duration} \
         --video-fps {fps} --video-synth-type moving_shapes \
         --prompt-input-tokens-mean 50 --num-dataset-entries 4 \
         --request-rate 2.0 --request-count 4 \
         --video-format {video_format} --video-codec {video_codec} \
         --workers-max {WORKERS_MAX}",
        h.mock.url
    ));

    assert_eq!(r.artifacts.request_count() as u32, 4);
    let inputs = r.artifacts.inputs();
    assert!(has_input_videos(&inputs));

    let videos = iter_video_details(&inputs);
    assert!(!videos.is_empty(), "No video content found in payloads");
    for details in &videos {
        assert_eq!(details.width, width);
        assert_eq!(details.height, height);
        assert!(approx_eq(details.fps, fps as f64));
        assert!(approx_eq(details.duration, duration));
        if check_fragmentation {
            assert!(
                !details.is_fragmented,
                "MP4 should use faststart, not fragmentation"
            );
        }
    }
}

#[tokio::test]
#[ignore = "requires ffmpeg for generation and ffprobe for stream inspection"]
async fn test_video_generation_parameters_webm() {
    video_generation_parameters("webm", "libvpx-vp9", false).await;
}

#[tokio::test]
#[ignore = "requires ffmpeg for generation and ffprobe for stream inspection"]
async fn test_video_generation_parameters_mp4() {
    video_generation_parameters("mp4", "libx264", true).await;
}

async fn video_with_audio_embeds_correct_stream(
    video_format: &str,
    video_codec: &str,
    expected_audio_codec: &str,
) {
    if cfg!(target_os = "windows") || !ffprobe_available() {
        return;
    }
    let (width, height, fps, duration) = (320, 240, 4, 2.0);

    let h = AIPerfHarness::new().await;
    let r = h.run(&format!(
        "--model {DEFAULT_MODEL} --url {} --endpoint-type chat \
         --video-width {width} --video-height {height} --video-duration {duration} \
         --video-fps {fps} --video-format {video_format} --video-codec {video_codec} \
         --video-audio-sample-rate 44.1 --video-audio-num-channels 1 \
         --prompt-input-tokens-mean 50 --num-dataset-entries 4 \
         --request-rate 2.0 --request-count 4 --workers-max {WORKERS_MAX}",
        h.mock.url
    ));

    assert_eq!(r.artifacts.request_count() as u32, 4);
    let inputs = r.artifacts.inputs();
    assert!(has_input_videos(&inputs));

    let videos = iter_video_details(&inputs);
    assert!(!videos.is_empty(), "No video content found in payloads");
    for details in &videos {
        assert_eq!(details.width, width);
        assert_eq!(details.height, height);
        assert!(approx_eq(details.fps, fps as f64));
        assert!(
            details.has_audio,
            "Expected audio stream in {video_format} video"
        );
        assert_eq!(details.audio_codec.as_deref(), Some(expected_audio_codec));
        assert_eq!(details.audio_channels, Some(1));
        assert_eq!(details.audio_sample_rate, Some(44100));
    }
}

#[tokio::test]
#[ignore = "requires ffmpeg for generation and ffprobe for stream inspection"]
async fn test_video_with_audio_embeds_correct_stream_webm() {
    video_with_audio_embeds_correct_stream("webm", "libvpx-vp9", "vorbis").await;
}

#[tokio::test]
#[ignore = "requires ffmpeg for generation and ffprobe for stream inspection"]
async fn test_video_with_audio_embeds_correct_stream_mp4() {
    video_with_audio_embeds_correct_stream("mp4", "libx264", "aac").await;
}

#[tokio::test]
#[ignore = "requires ffmpeg for generation and ffprobe for stream inspection"]
async fn test_video_without_audio_has_no_audio_stream() {
    if cfg!(target_os = "windows") || !ffprobe_available() {
        return;
    }
    let h = AIPerfHarness::new().await;
    let r = h.run(&format!(
        "--model {DEFAULT_MODEL} --url {} --endpoint-type chat \
         --video-width 320 --video-height 240 --video-duration 2.0 --video-fps 4 \
         --video-format webm --video-codec libvpx-vp9 \
         --prompt-input-tokens-mean 50 --num-dataset-entries 4 \
         --request-rate 2.0 --request-count 4 --workers-max {WORKERS_MAX}",
        h.mock.url
    ));

    assert_eq!(r.artifacts.request_count() as u32, 4);
    let inputs = r.artifacts.inputs();
    assert!(has_input_videos(&inputs));

    let videos = iter_video_details(&inputs);
    assert!(!videos.is_empty(), "No video content found in payloads");
    for details in &videos {
        assert!(
            !details.has_audio,
            "Video should not have audio when disabled"
        );
    }
}
