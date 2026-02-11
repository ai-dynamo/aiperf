<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# SGLang Video Generation

## Overview

This guide shows how to benchmark text-to-video generation APIs using SGLang and AIPerf. You'll learn how to set up the SGLang video generation server, create input prompts, run benchmarks, and analyze the results.

Video generation follows an **asynchronous job pattern**:
1. **Submit** - POST to `/v1/videos` with your prompt, receive a job ID
2. **Poll** - GET `/v1/videos/{id}` until status is `completed` or `failed`
3. **Download** - GET `/v1/videos/{id}/content` to retrieve the generated video

AIPerf handles this polling workflow automatically.

## References

For the most up-to-date information, please refer to the following resources:
- [SGLang Video Generation API](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/docs/openai_api.md)
- [SGLang Diffusion Installation Guide](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/docs/install.md)
- [SGLang CLI Reference](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/docs/cli.md)
- [OpenAI Videos API](https://platform.openai.com/docs/api-reference/videos)

## Supported Models

AIPerf supports any SGLang-compatible text-to-video model, including:

| Model | Model Path | Notes |
|-------|------------|-------|
| Wan2.1-T2V | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | Lightweight, good for testing |
| Wan2.1-T2V (14B) | `Wan-AI/Wan2.1-T2V-14B-Diffusers` | Higher quality, requires more VRAM |
| HunyuanVideo | `tencent/HunyuanVideo` | Tencent's video generation model |

## Setting Up the Server

### Option 1: Docker (Recommended)

**Export your Hugging Face token as an environment variable:**
```bash
export HF_TOKEN=<your-huggingface-token>
```

**Start the SGLang Docker container:**
```bash
docker run --gpus all \
    --shm-size 32g \
    -it \
    --rm \
    -p 30010:30010 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=$HF_TOKEN" \
    --ipc=host \
    lmsysorg/sglang:dev
```

> [!NOTE]
> The following steps are to be performed _inside_ the SGLang Docker container.

**Install the diffusion dependencies:**
```bash
uv pip install "sglang[diffusion]" --prerelease=allow --system
```

**Set the server arguments:**

> [!IMPORTANT]
> The following arguments set up the SGLang server to use Wan2.1-T2V-1.3B on port 30010.
> Adjust `--num-gpus`, `--ulysses-degree`, and `--ring-degree` based on your GPU configuration.

**Single GPU setup:**
```bash
SERVER_ARGS=(
    --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers
    --text-encoder-cpu-offload
    --pin-cpu-memory
    --num-gpus 1
    --port 30010
    --host 0.0.0.0
)
```

**Multi-GPU setup (4 GPUs with sequence parallelism):**
```bash
SERVER_ARGS=(
    --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers
    --text-encoder-cpu-offload
    --pin-cpu-memory
    --num-gpus 4
    --ulysses-degree 2
    --ring-degree 2
    --port 30010
    --host 0.0.0.0
)
```

**Start the SGLang server:**
```bash
sglang serve "${SERVER_ARGS[@]}"
```

**Wait until the server is ready** (watch the logs for the following message):
```
Uvicorn running on http://0.0.0.0:30010 (Press CTRL+C to quit)
```

### Option 2: Native Installation

**Install SGLang with diffusion support:**
```bash
pip install --upgrade pip
pip install uv
uv pip install "sglang[diffusion]" --prerelease=allow
```

**Start the server:**
```bash
sglang serve \
    --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --text-encoder-cpu-offload \
    --pin-cpu-memory \
    --num-gpus 1 \
    --port 30010 \
    --host 0.0.0.0
```

## Running the Benchmark

> [!NOTE]
> The following steps are to be performed on your local machine (_outside_ the SGLang Docker container).

### Basic Usage: Text-to-Video with Input File

**Create an input file with video prompts:**
```bash
cat > video_prompts.jsonl << 'EOF'
{"text": "A serene lake at sunset with mountains in the background"}
{"text": "A cat playing with a ball of yarn in a cozy living room"}
{"text": "A futuristic city with flying cars and neon lights"}
EOF
```

**Run the benchmark:**
```bash
aiperf profile \
    --model Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --tokenizer gpt2 \
    --url http://localhost:30010 \
    --endpoint-type video_generation \
    --input-file video_prompts.jsonl \
    --custom-dataset-type single_turn \
    --extra-inputs "size:1280x720" \
    --extra-inputs "seconds:4" \
    --concurrency 1 \
    --request-count 3
```

**Done!** This sends 3 requests to `http://localhost:30010/v1/videos` and polls until each video is complete.

**Sample Output (Successful Run):**
```
                                       NVIDIA AIPerf | Video Generation Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━┓
┃                            Metric ┃       avg ┃       min ┃       max ┃       p99 ┃       p90 ┃       p50 ┃     std ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━┩
│              Request Latency (ms) │ 45,234.56 │ 42,123.45 │ 48,567.89 │ 48,432.12 │ 47,654.32 │ 45,012.34 │ 2634.78 │
│    Input Sequence Length (tokens) │      8.33 │      7.00 │     10.00 │      9.98 │      9.80 │      8.00 │    1.25 │
│ Request Throughput (requests/sec) │      0.02 │         - │         - │         - │         - │         - │       - │
│          Request Count (requests) │      3.00 │         - │         - │         - │         - │         - │       - │
└───────────────────────────────────┴───────────┴───────────┴───────────┴───────────┴───────────┴───────────┴─────────┘
```

### Basic Usage: Text-to-Video with Synthetic Prompts

Generate videos using synthetic prompts with configurable token lengths:

```bash
aiperf profile \
    --model Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --tokenizer gpt2 \
    --url http://localhost:30010 \
    --endpoint-type video_generation \
    --extra-inputs "size:1280x720" \
    --extra-inputs "seconds:4" \
    --synthetic-input-tokens-mean 50 \
    --synthetic-input-tokens-stddev 10 \
    --concurrency 1 \
    --request-count 5
```

## Generation Parameters

Control video generation through `--extra-inputs`:

| Parameter | Description | Example |
|-----------|-------------|---------|
| `size` | Video resolution | `1280x720`, `720x1280`, `480x480` |
| `seconds` | Video duration in seconds | `4`, `8`, `12` |
| `seed` | Random seed for reproducibility | `42` |
| `num_inference_steps` | Diffusion denoising steps | `50` |
| `guidance_scale` | Classifier-free guidance scale | `7.5` |
| `negative_prompt` | Concepts to exclude | `"blurry, low quality"` |
| `fps` | Frames per second | `24` |
| `num_frames` | Total frames to generate | `48` |

**Video Download Option:**

Use `--download-video-content` to include video content download in the benchmark timing. When enabled, request latency includes the time to download the generated video from the server. By default, only generation time is measured.

```bash
aiperf profile \
    --model Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --tokenizer gpt2 \
    --url http://localhost:30010 \
    --endpoint-type video_generation \
    --input-file video_prompts.jsonl \
    --custom-dataset-type single_turn \
    --extra-inputs "size:1280x720" \
    --extra-inputs "seconds:4" \
    --download-video-content \
    --concurrency 1 \
    --request-count 3
```

**Example with advanced parameters:**
```bash
aiperf profile \
    --model Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --tokenizer gpt2 \
    --url http://localhost:30010 \
    --endpoint-type video_generation \
    --input-file video_prompts.jsonl \
    --custom-dataset-type single_turn \
    --extra-inputs "size:1280x720" \
    --extra-inputs "seconds:8" \
    --extra-inputs "seed:42" \
    --extra-inputs "guidance_scale:7.5" \
    --extra-inputs "num_inference_steps:50" \
    --concurrency 1 \
    --request-count 3
```

## Polling Configuration

AIPerf automatically handles polling for video generation. Configure polling behavior:

| Setting | Description | Default |
|---------|-------------|---------|
| `--request-timeout-seconds` | Maximum wait time before timeout | `21600` (6 hours) |
| `AIPERF_HTTP_VIDEO_POLL_INTERVAL` | Seconds between status checks (0.1-60) | `0.1` |

**Example with custom timeout and polling interval:**
```bash
# Set slower polling (0.5s) with 20 minute timeout
AIPERF_HTTP_VIDEO_POLL_INTERVAL=0.5 aiperf profile \
    --model Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --tokenizer gpt2 \
    --url http://localhost:30010 \
    --endpoint-type video_generation \
    --input-file video_prompts.jsonl \
    --custom-dataset-type single_turn \
    --extra-inputs "size:1280x720" \
    --extra-inputs "seconds:4" \
    --request-timeout-seconds 1200 \
    --concurrency 1 \
    --request-count 3
```

## Advanced Usage: Extracting Generated Videos

To extract and save the generated videos, use `--export-level raw` to capture the full response payloads.

**Run the benchmark with raw export:**
```bash
aiperf profile \
    --model Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --tokenizer gpt2 \
    --url http://localhost:30010 \
    --endpoint-type video_generation \
    --input-file video_prompts.jsonl \
    --custom-dataset-type single_turn \
    --extra-inputs "size:1280x720" \
    --extra-inputs "seconds:4" \
    --concurrency 1 \
    --request-count 3 \
    --export-level raw
```

**Download the generated videos:**

The response contains a URL to download the video. Copy the following script to `download_videos.py`:

```python
#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Download generated videos from AIPerf JSONL output file."""
import json
import os
from pathlib import Path
import sys
import urllib.request

# Read input file path
input_file = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    'artifacts/Wan-AI_Wan2.1-T2V-1.3B-Diffusers-openai-video_generation-concurrency1/profile_export_raw.jsonl'
)
output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('downloaded_videos')

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Process each line in the JSONL file
with open(input_file, 'r') as f:
    for line_num, line in enumerate(f, 1):
        record = json.loads(line)

        # Extract video URL from responses (look for the completed status response)
        for response in record.get('responses', []):
            response_data = json.loads(response.get('text', '{}'))

            # Check if this is a completed video response with a URL
            if response_data.get('status') == 'completed' and response_data.get('url'):
                video_url = response_data['url']
                video_id = response_data.get('id', f'video_{line_num}')

                # Download the video
                filename = output_dir / f"{video_id}.mp4"
                print(f"Downloading: {video_url}")

                try:
                    urllib.request.urlretrieve(video_url, filename)
                    print(f"Saved: {filename.resolve()}")
                except Exception as e:
                    print(f"Failed to download {video_id}: {e}")

print(f"\nVideos saved to: {output_dir.resolve()}")
```

**Run the script:**
```bash
python download_videos.py
```

**Output:**
```
Downloading: http://localhost:30010/v1/videos/video_abc123/content
Saved: /path/to/downloaded_videos/video_abc123.mp4
Downloading: http://localhost:30010/v1/videos/video_def456/content
Saved: /path/to/downloaded_videos/video_def456.mp4
Downloading: http://localhost:30010/v1/videos/video_ghi789/content
Saved: /path/to/downloaded_videos/video_ghi789.mp4

Videos saved to: /path/to/downloaded_videos
```

## Benchmark Scenarios

### Scenario 1: Throughput Testing

Test maximum throughput with multiple concurrent requests:

```bash
aiperf profile \
    --model Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --tokenizer gpt2 \
    --url http://localhost:30010 \
    --endpoint-type video_generation \
    --extra-inputs "size:720x480" \
    --extra-inputs "seconds:4" \
    --synthetic-input-tokens-mean 30 \
    --concurrency 4 \
    --request-count 20
```

### Scenario 2: Latency Testing

Test single-request latency for different video sizes:

```bash
# Short 4-second video
aiperf profile \
    --model Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --tokenizer gpt2 \
    --url http://localhost:30010 \
    --endpoint-type video_generation \
    --extra-inputs "size:1280x720" \
    --extra-inputs "seconds:4" \
    --synthetic-input-tokens-mean 50 \
    --concurrency 1 \
    --request-count 5

# Longer 8-second video
aiperf profile \
    --model Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --tokenizer gpt2 \
    --url http://localhost:30010 \
    --endpoint-type video_generation \
    --extra-inputs "size:1280x720" \
    --extra-inputs "seconds:8" \
    --synthetic-input-tokens-mean 50 \
    --concurrency 1 \
    --request-count 5
```

### Scenario 3: Quality vs Speed Trade-off

Compare generation quality at different inference step counts:

```bash
# Fast generation (fewer steps)
aiperf profile \
    --model Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --tokenizer gpt2 \
    --url http://localhost:30010 \
    --endpoint-type video_generation \
    --extra-inputs "size:1280x720" \
    --extra-inputs "seconds:4" \
    --extra-inputs "num_inference_steps:20" \
    --synthetic-input-tokens-mean 50 \
    --concurrency 1 \
    --request-count 5

# High quality (more steps)
aiperf profile \
    --model Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --tokenizer gpt2 \
    --url http://localhost:30010 \
    --endpoint-type video_generation \
    --extra-inputs "size:1280x720" \
    --extra-inputs "seconds:4" \
    --extra-inputs "num_inference_steps:100" \
    --synthetic-input-tokens-mean 50 \
    --concurrency 1 \
    --request-count 5
```

## Troubleshooting

### Connection Refused

If you see `Connection refused` errors:
1. Verify the SGLang server is running: `curl http://localhost:30010/health`
2. Check the port matches your server configuration
3. If using Docker, ensure port mapping is correct (`-p 30010:30010`)

### Timeout Errors

If requests time out during generation:
1. Increase the request timeout: `--request-timeout-seconds 1200`
2. Check server logs for errors
3. Reduce video resolution or duration for faster generation

### Out of Memory

If the server crashes with OOM errors:
1. Use a smaller model (e.g., Wan2.1-T2V-1.3B instead of 14B)
2. Reduce video resolution: `--extra-inputs "size:720x480"`
3. Enable CPU offloading: `--text-encoder-cpu-offload`
4. Reduce concurrency: `--concurrency 1`

### Model Not Found

If you see model loading errors:
1. Verify your Hugging Face token has access to the model
2. Check the model path is correct
3. Ensure sufficient disk space for model download

## Response Fields

The video generation API returns the following fields:

| Field | Description |
|-------|-------------|
| `id` | Unique video job identifier (mapped to `video_id` internally) |
| `object` | Object type, always `"video"` |
| `status` | Job status: `queued`, `in_progress`, `completed`, `failed` |
| `progress` | Completion percentage (0-100) |
| `url` | Download URL (only when `status=completed`) |
| `size` | Video resolution (e.g., `"1280x720"`) |
| `seconds` | Video duration (returned as string) |
| `quality` | Quality setting for the generated video |
| `model` | Model used for generation |
| `created_at` | Unix timestamp of job creation |
| `completed_at` | Unix timestamp of completion |
| `expires_at` | Unix timestamp when video assets expire |
| `inference_time_s` | Total generation time in seconds |
| `peak_memory_mb` | Peak GPU memory usage in MB |
| `error` | Error details if `status=failed` |

## Conclusion

You've successfully set up SGLang for video generation, run benchmarks with AIPerf, and learned how to download the generated videos. You can now experiment with different models, prompts, resolutions, and generation parameters to optimize your text-to-video workloads.

Key takeaways:
- Use `--endpoint-type video_generation`
- Control video parameters via `--extra-inputs`
- The transport handles polling automatically
- Use `--export-level raw` to capture full responses for video extraction

Now go forth and generate!
