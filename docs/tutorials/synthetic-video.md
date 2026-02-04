<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
- Multiple synthesis types (moving shapes, grid clock patterns)
- Configurable resolution, frame rate, and duration
- Hardware-accelerated encoding options (CPU and GPU codecs)
- Base64-encoded video output for API requests
- MP4 and WebM format support

## Basic Usage

### Example: Basic Video Generation

Generate videos with default settings (640x480, 4fps, 5 seconds):

```bash
aiperf profile \
    --model your-model-name \
    --endpoint-type chat \
    --endpoint /v1/chat/completions \
    --url localhost:8000 \
    --video-width 640 \
    --video-height 480 \
    --video-fps 4 \
    --video-duration 5.0 \
    --request-count 20
```

**Note:** Video generation is disabled by default (width=0, height=0). You must specify non-zero width and height to enable video generation.

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
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┓
┃                      Metric ┃     avg ┃     min ┃     max ┃     p99 ┃     p50 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━┩
│        Request Latency (ms) │ 2456.78 │ 2123.45 │ 2789.34 │ 2765.89 │ 2445.67 │
│    Time to First Token (ms) │  345.67 │  289.34 │  423.45 │  412.34 │  342.12 │
│    Inter Token Latency (ms) │   18.45 │   15.23 │   22.34 │   21.89 │   18.12 │
│ Output Token Count (tokens) │  150.00 │  145.00 │  158.00 │  157.45 │  150.00 │
│  Request Throughput (req/s) │    4.56 │       - │       - │       - │       - │
└─────────────────────────────┴─────────┴─────────┴─────────┴─────────┴─────────┘

JSON Export: artifacts/your-model-name-chat-concurrency1/profile_export_aiperf.json
```

## Configuration Options

### Video Dimensions

Control the resolution of generated videos:

```bash
aiperf profile \
    --model your-model-name \
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
    --model your-model-name \
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

AIPerf supports two built-in video patterns:

#### 1. Moving Shapes (Default)

Generates videos with animated geometric shapes moving across the screen:

```bash
aiperf profile \
    --model your-model-name \
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
    --model your-model-name \
    --endpoint-type chat \
    --url localhost:8000 \
    --video-width 640 \
    --video-height 480 \
    --video-synth-type grid_clock \
    --request-count 20
```

Features:
- Grid overlay
- Animated rotating elements
- Gray background
- Visual timing markers

## Advanced Configuration

### Video Codec Selection

Choose encoding codec based on your hardware and requirements:

#### CPU Encoding (Default)

```bash
aiperf profile \
    --model your-model-name \
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
    --model your-model-name \
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
    --model your-model-name \
    --endpoint-type chat \
    --url localhost:8000 \
    --video-width 640 \
    --video-height 480 \
    --video-batch-size 2 \
    --request-count 10
```

## Example Workflows

### Example 1: Low-Resolution Video Understanding

Benchmark with small, low-framerate videos:

```bash
aiperf profile \
    --model your-model-name \
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
    --model your-model-name \
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
    --model your-model-name \
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

### Example 4: Rapid Short Clips

Test with many short video clips:

```bash
aiperf profile \
    --model your-model-name \
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
    --model your-model-name \
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
    --model your-model-name \
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
Error: FFmpeg is required for video generation but was not found
Install with: sudo apt install ffmpeg
```

Follow the installation instructions in the Prerequisites section.

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

## Summary

The synthetic video generation feature enables comprehensive benchmarking of video understanding models with:

- Flexible video parameters (resolution, frame rate, duration)
- Multiple synthesis patterns for variety
- Hardware-accelerated encoding options
- Easy integration with multimodal APIs

Use synthetic videos to test your model's performance across different video characteristics without requiring large video datasets.

