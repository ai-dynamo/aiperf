<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Profile Vision Language Models with AIPerf

AIPerf supports benchmarking Vision Language Models (VLMs) that process both text and images.

This guide covers profiling vision models using OpenAI-compatible chat completions endpoints with vLLM.

---

## Start a vLLM Server

Launch a vLLM server with a vision language model:

```bash
docker pull vllm/vllm-openai:latest
docker run --gpus all -p 8000:8000 vllm/vllm-openai:latest \
  --model Qwen/Qwen2-VL-2B-Instruct
```

Verify the server is ready:
```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2-VL-2B-Instruct",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 10
  }' | jq
```

---

## Profile with Synthetic Images

AIPerf can generate synthetic images for benchmarking.

```bash
aiperf profile \
    --model Qwen/Qwen2-VL-2B-Instruct \
    --endpoint-type chat \
    --image-width-mean 512 \
    --image-height-mean 512 \
    --synthetic-input-tokens-mean 100 \
    --streaming \
    --url localhost:8000 \
    --request-count 20 \
    --concurrency 4
```

---

## Profile with Custom Input File

Create a JSONL file with text prompts and image URLs:

```bash
cat <<EOF > inputs.jsonl
{"texts": ["Describe this image in detail."], "images": ["https://picsum.photos/512/512?random=1"]}
{"texts": ["What objects are visible in this image?"], "images": ["https://picsum.photos/512/512?random=2"]}
{"texts": ["Analyze the composition of this photo."], "images": ["https://picsum.photos/512/512?random=3"]}
{"texts": ["What is the main subject of this image?"], "images": ["https://picsum.photos/512/512?random=4"]}
{"texts": ["Provide a caption for this image."], "images": ["https://picsum.photos/512/512?random=5"]}
EOF
```

Run AIPerf using the custom input file:
```bash
aiperf profile \
    --model Qwen/Qwen2-VL-2B-Instruct \
    --endpoint-type chat \
    --input-file inputs.jsonl \
    --custom-dataset-type single_turn \
    --streaming \
    --url localhost:8000 \
    --request-count 5
```
