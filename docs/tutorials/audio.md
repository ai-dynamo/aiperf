<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Profile Audio Language Models with AIPerf

AIPerf supports benchmarking Audio Language Models that process audio inputs with optional text prompts.

This guide covers profiling audio models using OpenAI-compatible chat completions endpoints with vLLM.

---

## Start a vLLM Server

Launch the vLLM server with Qwen2-Audio-7B-Instruct:

<!-- setup-vllm-audio-openai-endpoint-server -->
```bash
docker pull vllm/vllm-openai:latest
docker run --gpus all -p 8000:8000 vllm/vllm-openai:latest \
  --model Qwen/Qwen2-Audio-7B-Instruct \
  --trust-remote-code
```
<!-- /setup-vllm-audio-openai-endpoint-server -->


Verify the server is ready:

<!-- health-check-vllm-audio-openai-endpoint-server -->
```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2-Audio-7B-Instruct",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 10
  }' | jq
```
<!-- /health-check-vllm-audio-openai-endpoint-server -->

---

## Profile with Synthetic Audio

AIPerf can generate synthetic audio for benchmarking:

<!-- aiperf-run-vllm-audio-openai-endpoint-server -->
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
<!-- /aiperf-run-vllm-audio-openai-endpoint-server -->

To add text prompts alongside audio, include `--synthetic-input-tokens-mean 100`

## Profile with Custom Input File

Create a JSONL file with audio data and optional text prompts.

<!-- aiperf-run-vllm-audio-openai-endpoint-server -->
```bash
cat <<EOF > inputs.jsonl
{"texts": ["Transcribe this audio."], "audios": ["wav,UklGRiIFAABXQVZFZm10IBAAAAABAAEAgD4AAAB9AAACABAAZGF0Yf4EAAD..."]}
{"texts": ["What is being said in this recording?"], "audios": ["mp3,SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU4Ljc2LjEwMAAAAAAAAAAA..."]}
{"texts": ["Summarize the main points from this audio."], "audios": ["wav,UklGRooGAABXQVZFZm10IBAAAAABAAEAgD4AAAB9AAACABAAZGF0YWY..."]}
EOF
```
<!-- /aiperf-run-vllm-audio-openai-endpoint-server -->

The audio data format is: `{format},{base64_encoded_audio_data}` where:
- `format`: Either `wav` or `mp3`
- `base64_encoded_audio_data`: Base64-encoded audio file content

Run AIPerf using the custom input file:

<!-- aiperf-run-vllm-audio-openai-endpoint-server -->
```bash
aiperf profile \
    --model Qwen/Qwen2-Audio-7B-Instruct \
    --endpoint-type chat \
    --input-file inputs.jsonl \
    --custom-dataset-type single_turn \
    --streaming \
    --url localhost:8000 \
    --request-count 3
```
<!-- /aiperf-run-vllm-audio-openai-endpoint-server -->