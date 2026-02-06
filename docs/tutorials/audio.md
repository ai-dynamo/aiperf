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

AIPerf can automatically load and encode audio files from local paths:

```bash
cat <<EOF > inputs.jsonl
{"texts": ["Transcribe this audio."], "audios": ["/path/to/audio1.wav"]}
{"texts": ["What is being said in this recording?"], "audios": ["/path/to/audio2.mp3"]}
{"texts": ["Summarize the main points from this audio."], "audios": ["/path/to/audio3.wav"]}
EOF
```

AIPerf will automatically:
- Load the audio files from the specified paths
- Convert them to base64 format
- Send them to the model endpoint

Run AIPerf with the file path input:

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

<!-- CI-ONLY: Hidden test for file path loading using existing vLLM server
aiperf-run-vllm-audio-openai-endpoint-server
```bash
cat <<EOF > inputs_filepaths.jsonl
{"texts": ["Transcribe this."], "audios": ["/fixtures/audio/test_audio_1s.wav"]}
{"texts": ["What is said?"], "audios": ["/fixtures/audio/test_audio_2.wav"]}
{"texts": ["Summarize."], "audios": ["/fixtures/audio/test_audio_3.wav"]}
EOF
aiperf profile \
    --model Qwen/Qwen2-Audio-7B-Instruct \
    --endpoint-type chat \
    --input-file inputs_filepaths.jsonl \
    --custom-dataset-type single_turn \
    --streaming \
    --url localhost:8000 \
    --request-count 3
```
-->