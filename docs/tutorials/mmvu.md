---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Profile with MMVU Dataset
---

# Profile with MMVU Dataset

AIPerf supports benchmarking using the MMVU dataset, an expert-level video understanding
benchmark that tests multi-discipline reasoning over video content. Each sample contains a
video URL and a question (multiple-choice or open-ended) that requires watching the video
to answer.

This guide covers profiling OpenAI-compatible video language models using the MMVU public
dataset.

---

## Start a vLLM Server

Launch a vLLM server with a video-capable vision language model:

```bash
docker pull vllm/vllm-openai:latest
docker run --gpus all -p 8000:8000 vllm/vllm-openai:latest \
  --model Qwen/Qwen2-VL-7B-Instruct
```

Verify the server is ready:
```bash
curl -s localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2-VL-7B-Instruct","messages":[{"role":"user","content":"test"}],"max_tokens":1}'
```

---

## Profile with MMVU Dataset

AIPerf loads the MMVU dataset from HuggingFace, combines each question with its
multiple-choice options, attaches the video URL, and sends each pair as a single-turn
video request. The prompt format matches vLLM's own MMVU benchmark format.

{/* aiperf-run-vllm-video-openai-endpoint-server */}
```bash
aiperf profile \
    --model Qwen/Qwen2-VL-7B-Instruct \
    --endpoint-type chat \
    --streaming \
    --url localhost:8000 \
    --public-dataset mmvu \
    --request-count 10 \
    --concurrency 4
```
{/* /aiperf-run-vllm-video-openai-endpoint-server */}

---

## Notes

- The `video` column in MMVU contains HTTPS URLs pointing to `.mp4` files hosted on
  HuggingFace. AIPerf passes these URLs directly to the model server, which fetches
  the video during inference.
- For multiple-choice questions, choices are appended to the question in the format
  `A.option B.option ...`. Open-ended questions use the question text only.
- The dataset has a `validation` split with samples spanning multiple academic disciplines
  (Art, Science, Engineering, etc.).
