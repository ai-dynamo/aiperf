<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Synthetic Video Generation

AIPerf supports synthetic video generation for benchmarking multimodal models that process video inputs. This feature allows you to generate videos with different patterns, resolutions, frame rates, and durations to simulate various video understanding workloads.

## Prerequisites

Video generation requires FFmpeg to be installed on your system.

### Installing FFmpeg

**Ubuntu/Debian:**
```bash
sudo apt update && sudo apt install ffmpeg
```

**macOS (with Homebrew):**
```bash
brew install ffmpeg
```

**Fedora/RHEL/CentOS:**
```bash
sudo dnf install ffmpeg
```

**Windows (with Chocolatey):**
```bash
choco install ffmpeg
```

## Overview

The synthetic video feature provides:
- Multiple synthesis types (moving shapes, grid clock, noise patterns)
- Configurable resolution, frame rate, and duration
- Hardware-accelerated encoding options (CPU and GPU codecs)
- Embedded synthetic audio tracks for video+audio multimodal benchmarking
- Base64-encoded video output for API requests
- MP4 and WebM format support

## Basic Usage

### Example: Basic Video Generation

Generate videos at 640x480 with default temporal settings (4 fps, 5 seconds):

```bash
aiperf profile \
    --model Qwen/Qwen2-VL-2B-Instruct \
    --endpoint-type chat \
    --endpoint /v1/chat/completions \
    --url localhost:8000 \
    --video-width 640 \
    --video-height 480 \
    --video-fps 4 \
    --video-duration 5.0 \
    --request-count 20
```

**Note:** Video generation is disabled by default (width and height are unset). You must specify both width and height to enable video generation.

**Sample Output (Successful Run):**
```
INFO     Starting AIPerf System
INFO     Generating synthetic videos (640x480 px, 4 fps, 5.0s duration)
INFO     Video codec: libvpx-vp9 (format: webm)
INFO     AIPerf System is PROFILING

Profiling: 20/20 |████████████████████████| 100% [01:45<00:00]

INFO     Benchmark completed successfully
INFO     Results saved to: artifacts/your-model-name-chat-concurrency1/

            NVIDIA AIPerf | LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┓
┃                      Metric ┃     avg ┃     min ┃     max ┃     p99 ┃     p90 ┃     p50 ┃     std ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━┩
│        Request Latency (ms) │ 2456.78 │ 2123.45 │ 2789.34 │ 2765.89 │ 2698.12 │ 2445.67 │  198.34 │
│    Time to First Token (ms) │  345.67 │  289.34 │  423.45 │  412.34 │  398.56 │  342.12 │   38.21 │
│    Inter Token Latency (ms) │   18.45 │   15.23 │   22.34 │   21.89 │   20.78 │   18.12 │    2.15 │
│ Output Token Count (tokens) │  150.00 │  145.00 │  158.00 │  157.45 │  156.12 │  150.00 │    3.87 │
│ Request Throughput (requests/sec) │    4.56 │     N/A │     N/A │     N/A │     N/A │     N/A │     N/A │
└─────────────────────────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┘

JSON Export: artifacts/your-model-name-chat-concurrency1/profile_export_aiperf.json
```

## Configuration Options

### Video Dimensions

Control the resolution of generated videos:

```bash
aiperf profile \
    --model Qwen/Qwen2-VL-2B-Instruct \
    --endpoint-type chat \
    --url localhost:8000 \
    --video-width 1920 \
    --video-height 1080 \
    --request-count 10
```

### Frame Rate and Duration

Adjust temporal properties:

```bash
aiperf profile \
    --model Qwen/Qwen2-VL-2B-Instruct \
    --endpoint-type chat \
    --url localhost:8000 \
    --video-width 640 \
    --video-height 480 \
    --video-fps 8 \
    --video-duration 10.0 \
    --request-count 15
```

**Parameters:**
- `--video-fps`: Frames per second (default: 4, recommended for models like Cosmos)
- `--video-duration`: Clip duration in seconds (default: 5.0)

### Synthesis Types

AIPerf supports three built-in video patterns:

#### 1. Moving Shapes (Default)

Generates videos with animated geometric shapes moving across the screen:

```bash
aiperf profile \
    --model Qwen/Qwen2-VL-2B-Instruct \
    --endpoint-type chat \
    --url localhost:8000 \
    --video-width 640 \
    --video-height 480 \
    --video-synth-type moving_shapes \
    --request-count 20
```

Features:
- Multiple colored shapes (circles and rectangles)
- Smooth motion patterns
- Wrapping at screen edges
- Black background

#### 2. Grid Clock

Generates videos with a grid pattern and clock-like animation:

```bash
aiperf profile \
    --model Qwen/Qwen2-VL-2B-Instruct \
    --endpoint-type chat \
    --url localhost:8000 \
    --video-width 640 \
    --video-height 480 \
    --video-synth-type grid_clock \
    --request-count 20
```

Features:
- Grid overlay
- Animated clock hands (hour and minute)
- Dark gray background
- Frame number overlay

#### 3. Noise

Generates videos with random noise pixels in each frame:

```bash
aiperf profile \
    --model Qwen/Qwen2-VL-2B-Instruct \
    --endpoint-type chat \
    --url localhost:8000 \
    --video-width 640 \
    --video-height 480 \
    --video-synth-type noise \
    --request-count 20
```

Features:
- Random RGB pixel values per frame
- Deterministic output via seeded RNG
- Maximum entropy content for codec stress testing

## Advanced Configuration

### Video Codec Selection

Choose encoding codec based on your hardware and requirements:

#### CPU Encoding (Default)

```bash
aiperf profile \
    --model Qwen/Qwen2-VL-2B-Instruct \
    --endpoint-type chat \
    --url localhost:8000 \
    --video-width 640 \
    --video-height 480 \
    --video-codec libvpx-vp9 \
    --video-format webm \
    --request-count 20
```

**Available CPU Codecs:**
- `libvpx-vp9`: VP9 encoding, BSD-licensed (default, WebM format)
- `libx264`: H.264 encoding, GPL-licensed, widely compatible (MP4 format)
- `libx265`: H.265 encoding, GPL-licensed, smaller file sizes, slower encoding (MP4 format)

#### GPU Encoding (NVIDIA)

For faster encoding with NVIDIA GPUs:

```bash
aiperf profile \
    --model Qwen/Qwen2-VL-2B-Instruct \
    --endpoint-type chat \
    --url localhost:8000 \
    --video-width 1920 \
    --video-height 1080 \
    --video-codec h264_nvenc \
    --request-count 50
```

**Available NVIDIA GPU Codecs:**
- `h264_nvenc`: H.264 GPU encoding
- `hevc_nvenc`: H.265 GPU encoding, smaller files

### Batch Size

Control the number of videos per request:

```bash
aiperf profile \
    --model Qwen/Qwen2-VL-2B-Instruct \
    --endpoint-type chat \
    --url localhost:8000 \
    --video-width 640 \
    --video-height 480 \
    --video-batch-size 2 \
    --request-count 10
```

### Embedded Audio Track

AIPerf can embed a synthetic audio track into generated videos for benchmarking multimodal models that process video+audio inputs together. When enabled, a Gaussian noise audio signal matching the video duration is muxed into each video file via FFmpeg.

Audio embedding is disabled by default to maintain backward compatibility and minimize file size for video-only workloads.

#### Enabling Audio

Set `--video-audio-num-channels` to 1 (mono) or 2 (stereo) to embed an audio track:

```bash
aiperf profile \
    --model Qwen/Qwen2-VL-2B-Instruct \
    --endpoint-type chat \
    --url localhost:8000 \
    --video-width 640 \
    --video-height 480 \
    --video-audio-num-channels 1 \
    --request-count 20
```

This generates videos with a mono, 44100 Hz audio track using an auto-selected codec (libvorbis for WebM, aac for MP4).

#### Audio Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `--video-audio-num-channels` | `int` | `0` | 0 = disabled, 1 = mono, 2 = stereo |
| `--video-audio-sample-rate` | `int` | `44100` | Sample rate in Hz (8000-96000) |
| `--video-audio-codec` | `string` | auto | Audio codec (`aac`, `libvorbis`, `libopus`) |
| `--video-audio-depth` | `int` | `16` | Bit depth per sample (8, 16, 24, or 32) |

#### Audio Codec Selection

When `--video-audio-codec` is not specified, the codec is automatically selected based on the video format:

| Video Format | Auto-Selected Audio Codec |
|---|---|
| WebM | `libvorbis` (Vorbis) |
| MP4 | `aac` (AAC) |

You can override the auto-selection with an explicit codec:

```bash
aiperf profile \
    --model Qwen/Qwen2-VL-2B-Instruct \
    --endpoint-type chat \
    --url localhost:8000 \
    --video-width 640 \
    --video-height 480 \
    --video-format webm \
    --video-audio-num-channels 1 \
    --video-audio-codec libopus \
    --request-count 20
```

#### Stereo Audio with Custom Sample Rate

```bash
aiperf profile \
    --model Qwen/Qwen2-VL-2B-Instruct \
    --endpoint-type chat \
    --url localhost:8000 \
    --video-width 640 \
    --video-height 480 \
    --video-audio-num-channels 2 \
    --video-audio-sample-rate 48000 \
    --request-count 20
```

#### How It Works

1. A Gaussian noise audio signal is generated matching the video duration
2. The audio is encoded as 16-bit PCM WAV
3. FFmpeg muxes the video and audio streams together using `-shortest` to ensure duration alignment
4. The audio codec converts the WAV data to the target format (AAC, Vorbis, or Opus)
5. The resulting video+audio file is base64-encoded for API requests

The audio generation uses a deterministic RNG seed (`dataset.video.audio`), so videos with audio are reproducible across runs when using `--random-seed`.

#### Audio Size Impact

Factors affecting audio contribution to file size:
- **Sample rate**: 48000 Hz produces ~9% more data than 44100 Hz
- **Channels**: Stereo (2) doubles audio data compared to mono (1)
- **Codec**: Vorbis and Opus provide better compression than AAC at lower bitrates
- **Duration**: Audio size scales linearly with video duration

For most benchmarking scenarios, the audio track adds minimal overhead compared to the video stream.

## Example Workflows

### Example 1: Low-Resolution Video Understanding

Benchmark with small, low-framerate videos:

```bash
aiperf profile \
    --model Qwen/Qwen2-VL-2B-Instruct \
    --endpoint-type chat \
    --url localhost:8000 \
    --video-width 320 \
    --video-height 240 \
    --video-fps 2 \
    --video-duration 3.0 \
    --video-synth-type moving_shapes \
    --concurrency 4 \
    --request-count 50
```

**Use case:** Testing lightweight video processing or mobile-optimized models.

### Example 2: HD Video Benchmarking

Test with high-resolution, longer videos:

```bash
aiperf profile \
    --model Qwen/Qwen2-VL-2B-Instruct \
    --endpoint-type chat \
    --url localhost:8000 \
    --video-width 1920 \
    --video-height 1080 \
    --video-fps 8 \
    --video-duration 10.0 \
    --video-codec h264_nvenc \
    --concurrency 2 \
    --request-count 20
```

**Use case:** Stress testing with high-quality video inputs.

### Example 3: Mixed Text and Video

Combine video with text prompts for multimodal testing:

```bash
aiperf profile \
    --model Qwen/Qwen2-VL-2B-Instruct \
    --endpoint-type chat \
    --url localhost:8000 \
    --video-width 640 \
    --video-height 480 \
    --video-fps 4 \
    --video-duration 5.0 \
    --synthetic-input-tokens-mean 100 \
    --synthetic-input-tokens-stddev 20 \
    --output-tokens-mean 50 \
    --concurrency 8 \
    --request-count 100
```

**Use case:** Simulating video question-answering or video captioning workloads.

### Example 4: Video + Audio Multimodal

Benchmark models that process both video and audio streams:

```bash
aiperf profile \
    --model Qwen/Qwen2-VL-2B-Instruct \
    --endpoint-type chat \
    --url localhost:8000 \
    --video-width 640 \
    --video-height 480 \
    --video-fps 4 \
    --video-duration 5.0 \
    --video-audio-num-channels 1 \
    --video-audio-sample-rate 16000 \
    --concurrency 4 \
    --request-count 50
```

**Use case:** Testing video+audio understanding models (e.g., video QA with spoken audio, meeting transcription with video context).

### Example 5: Video + Audio with MP4 and Stereo

Test with MP4 format and stereo audio for maximum compatibility:

```bash
aiperf profile \
    --model Qwen/Qwen2-VL-2B-Instruct \
    --endpoint-type chat \
    --url localhost:8000 \
    --video-width 1280 \
    --video-height 720 \
    --video-fps 8 \
    --video-duration 10.0 \
    --video-format mp4 \
    --video-codec libx264 \
    --video-audio-num-channels 2 \
    --video-audio-sample-rate 44100 \
    --concurrency 2 \
    --request-count 20
```

**Use case:** Simulating real-world video files with stereo audio tracks for production-like multimodal workloads.

### Example 6: Rapid Short Clips

Test with many short video clips:

```bash
aiperf profile \
    --model Qwen/Qwen2-VL-2B-Instruct \
    --endpoint-type chat \
    --url localhost:8000 \
    --video-width 640 \
    --video-height 360 \
    --video-fps 4 \
    --video-duration 2.0 \
    --video-synth-type grid_clock \
    --concurrency 16 \
    --request-count 200
```

**Use case:** Testing throughput with brief video clips.

## Format and Output

### Video Format

AIPerf supports both **WebM** (default) and **MP4** formats:

**WebM format (default):**
```bash
aiperf profile \
    --model Qwen/Qwen2-VL-2B-Instruct \
    --endpoint-type chat \
    --url localhost:8000 \
    --video-width 640 \
    --video-height 480 \
    --video-format webm \
    --video-codec libvpx-vp9 \
    --request-count 20
```

**MP4 format:**
```bash
aiperf profile \
    --model Qwen/Qwen2-VL-2B-Instruct \
    --endpoint-type chat \
    --url localhost:8000 \
    --video-width 640 \
    --video-height 480 \
    --video-format mp4 \
    --video-codec libx264 \
    --request-count 20
```

### Data Encoding

Generated videos are automatically:
1. Encoded using the specified codec
2. Converted to base64 strings
3. Embedded in API request payloads

This allows seamless integration with vision-language model APIs that accept base64-encoded video content.

## Performance Considerations

### Encoding Performance

- **CPU codecs** (`libvpx-vp9`, `libx264`, `libx265`): Slower but universally available
- **GPU codecs** (`h264_nvenc`, `hevc_nvenc`): Much faster, requires NVIDIA GPU
- Higher resolution and frame rates increase encoding time

### Video Size Impact

Factors affecting video file size:
- **Resolution**: Higher dimensions = larger files
- **Duration**: Longer videos = larger files
- **Frame rate**: More frames = larger files
- **Codec**: H.265/HEVC produces smaller files than H.264

### Recommendations

1. **For high-throughput testing**: Use lower resolutions (320x240 or 640x480) and GPU encoding
2. **For quality testing**: Use higher resolutions (1920x1080) with appropriate concurrency limits
3. **For API payload testing**: Match your production video specifications
4. **For development**: Start with small dimensions and short durations

## Troubleshooting

### FFmpeg Not Found

If you see an error about FFmpeg not being installed:

```
RuntimeError: FFmpeg binary not found. Please install FFmpeg:

  Recommended: sudo apt update && sudo apt install ffmpeg

  Alternative: conda install -c conda-forge ffmpeg

After installation, restart your terminal and try again.
```

Follow the installation instructions in the [Prerequisites](#prerequisites) section.

### GPU Codec Not Available

If NVIDIA GPU codecs fail:

```
Error: Encoder 'h264_nvenc' not found
```

Solutions:
1. Verify NVIDIA GPU is available: `nvidia-smi`
2. Check FFmpeg was compiled with NVENC support: `ffmpeg -encoders | grep nvenc`
3. Fall back to CPU codec: `--video-codec libvpx-vp9 --video-format webm` or `--video-codec libx264 --video-format mp4`

### Out of Memory

For high-resolution or long-duration videos:
1. Reduce `--video-width` and `--video-height`
2. Decrease `--video-duration`
3. Lower `--concurrency`

## CLI Reference

All video-related parameters at a glance:

### Video Parameters

| Parameter | Default | Description |
|---|---|---|
| `--video-width` | `None` | Frame width in pixels (must pair with height) |
| `--video-height` | `None` | Frame height in pixels (must pair with width) |
| `--video-fps` | `4` | Frames per second |
| `--video-duration` | `5.0` | Clip duration in seconds |
| `--video-batch-size` | `1` | Videos per request |
| `--video-synth-type` | `moving_shapes` | Synthesis pattern (`moving_shapes`, `grid_clock`, `noise`) |
| `--video-format` | `webm` | Container format (`webm`, `mp4`) |
| `--video-codec` | `libvpx-vp9` | Video codec (any FFmpeg-supported codec) |

### Audio Parameters

| Parameter | Default | Description |
|---|---|---|
| `--video-audio-num-channels` | `0` | 0 = disabled, 1 = mono, 2 = stereo |
| `--video-audio-sample-rate` | `44100` | Sample rate in Hz (8000-96000) |
| `--video-audio-codec` | auto | Audio codec (`aac`, `libvorbis`, `libopus`) |
| `--video-audio-depth` | `16` | Bit depth per sample (8, 16, 24, or 32) |

## Summary

The synthetic video generation feature enables comprehensive benchmarking of video understanding models with:

- Flexible video parameters (resolution, frame rate, duration)
- Multiple synthesis patterns for variety
- Hardware-accelerated encoding options
- Optional embedded audio tracks for video+audio multimodal workloads
- Easy integration with multimodal APIs

Use synthetic videos to test your model's performance across different video characteristics without requiring large video datasets.
