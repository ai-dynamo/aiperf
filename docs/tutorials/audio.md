<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Profile Audio Language Models with AIPerf

AIPerf supports benchmarking Audio Language Models that process audio inputs with optional text prompts.

This guide covers profiling audio models using OpenAI-compatible chat completions endpoints with vLLM.

---

## Start a vLLM Server

Launch a vLLM server with Qwen2-Audio-7B-Instruct:

```bash
# Using vLLM directly
vllm serve Qwen/Qwen2-Audio-7B-Instruct \
  --port 8000 \
  --trust-remote-code \
  --max-model-len 4096 \
  --limit-mm-per-prompt audio=2

# Or using Docker
docker pull vllm/vllm-openai:latest
docker run --gpus all -p 8000:8000 vllm/vllm-openai:latest \
  --model Qwen/Qwen2-Audio-7B-Instruct \
  --trust-remote-code \
  --max-model-len 4096 \
  --limit-mm-per-prompt audio=2
```

Verify the server is ready:
```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2-Audio-7B-Instruct",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 10
  }' | jq
```

---

## Profile with Synthetic Audio

AIPerf can generate synthetic audio for benchmarking.

### Audio-Only Benchmark

Profile with audio inputs only (no text prompts):

```bash
aiperf profile \
    --model Qwen/Qwen2-Audio-7B-Instruct \
    --endpoint-type chat \
    --audio-length-mean 5.0 \
    --audio-format wav \
    --audio-sample-rates 16 \
    --streaming \
    --url localhost:8000 \
    --request-count 20 \
    --concurrency 4
```

### Audio with Text Prompts

Add text prompts to provide context or instructions for the audio:

```bash
aiperf profile \
    --model Qwen/Qwen2-Audio-7B-Instruct \
    --endpoint-type chat \
    --audio-length-mean 5.0 \
    --audio-format wav \
    --audio-sample-rates 16 \
    --synthetic-input-tokens-mean 100 \
    --streaming \
    --url localhost:8000 \
    --request-count 20 \
    --concurrency 4
```

### Audio Configuration Options

AIPerf provides several options to control synthetic audio generation:

- `--audio-length-mean`: Mean audio duration in seconds (default: 10.0)
- `--audio-length-stddev`: Standard deviation of audio duration (default: 0.0)
- `--audio-format`: Audio format - `wav` or `mp3` (default: wav)
- `--audio-sample-rates`: List of sample rates in kHz to randomly select from (default: 16)
- `--audio-depths`: List of bit depths to randomly select from (default: 16)
- `--audio-num-channels`: Number of audio channels - 1 (mono) or 2 (stereo) (default: 1)
- `--audio-batch-size`: Number of audio inputs per request (default: 1)
