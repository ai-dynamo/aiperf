<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Profile Vision Language Models with AIPerf

AIPerf supports benchmarking Vision Language Models (VLMs) that process both text and images.

This guide covers profiling vision models using OpenAI-compatible chat completions endpoints with vLLM.

---

## Start a vLLM Server

Launch a vLLM server with a vision language model:

<!-- setup-vllm-vision-openai-endpoint-server -->
```bash
docker pull vllm/vllm-openai:latest
docker run --gpus all -p 8000:8000 vllm/vllm-openai:latest \
  --model Qwen/Qwen2-VL-2B-Instruct
```
<!-- /setup-vllm-vision-openai-endpoint-server -->

Verify the server is ready:

<!-- health-check-vllm-vision-openai-endpoint-server -->
```bash
timeout 900 bash -c 'while [ "$(curl -s -o /dev/null -w "%{http_code}" localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"Qwen/Qwen2-VL-2B-Instruct\",\"messages\":[{\"role\":\"user\",\"content\":\"test\"}],\"max_tokens\":1}")" != "200" ]; do sleep 2; done' || { echo "vLLM not ready after 15min"; exit 1; }
```
<!-- /health-check-vllm-vision-openai-endpoint-server -->

---

## Profile with Synthetic Images

AIPerf can generate synthetic images for benchmarking.

<!-- aiperf-run-vllm-vision-openai-endpoint-server -->
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
<!-- /aiperf-run-vllm-vision-openai-endpoint-server -->

**Sample Output (Successful Run):**
```
INFO     Starting AIPerf System
INFO     Generating synthetic images (512x512 px)
INFO     AIPerf System is PROFILING

Profiling: 20/20 |████████████████████████| 100% [01:45<00:00]

INFO     Benchmark completed successfully
INFO     Results saved to: artifacts/Qwen_Qwen2-VL-2B-Instruct-chat-concurrency4/

            NVIDIA AIPerf | LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┓
┃                      Metric ┃     avg ┃     min ┃     max ┃     p99 ┃     p50 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━┩
│        Request Latency (ms) │ 2345.67 │ 1890.34 │ 2987.12 │ 2923.45 │ 2312.89 │
│    Time to First Token (ms) │  456.78 │  378.90 │  598.45 │  578.23 │  445.67 │
│    Inter Token Latency (ms) │   18.90 │   14.56 │   25.34 │   24.12 │   18.45 │
│ Output Token Count (tokens) │  150.00 │  120.00 │  180.00 │  178.00 │  148.00 │
│  Request Throughput (req/s) │    4.89 │       - │       - │       - │       - │
└─────────────────────────────┴─────────┴─────────┴─────────┴─────────┴─────────┘

JSON Export: artifacts/Qwen_Qwen2-VL-2B-Instruct-chat-concurrency4/profile_export_aiperf.json
```

---

## Profile with Custom Input File

Create a JSONL file with text prompts and image URLs:

<!-- aiperf-run-vllm-vision-openai-endpoint-server -->
```bash
cat <<EOF > inputs.jsonl
{"texts": ["Describe this image in detail."], "images": ["https://picsum.photos/512/512?random=1"]}
{"texts": ["What objects are visible in this image?"], "images": ["https://picsum.photos/512/512?random=2"]}
{"texts": ["Analyze the composition of this photo."], "images": ["https://picsum.photos/512/512?random=3"]}
{"texts": ["What is the main subject of this image?"], "images": ["https://picsum.photos/512/512?random=4"]}
{"texts": ["Provide a caption for this image."], "images": ["https://picsum.photos/512/512?random=5"]}
EOF
```
<!-- /aiperf-run-vllm-vision-openai-endpoint-server -->

Run AIPerf using the custom input file:

<!-- aiperf-run-vllm-vision-openai-endpoint-server -->
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
<!-- /aiperf-run-vllm-vision-openai-endpoint-server -->

**Sample Output (Successful Run):**
```
INFO     Starting AIPerf System
INFO     Loaded 5 entries from inputs.jsonl
INFO     Using single_turn dataset type with custom images
INFO     AIPerf System is PROFILING

Profiling: 5/5 |████████████████████████| 100% [00:25<00:00]

INFO     Benchmark completed successfully
INFO     Results saved to: artifacts/Qwen_Qwen2-VL-2B-Instruct-chat-concurrency1/

            NVIDIA AIPerf | LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┓
┃                      Metric ┃     avg ┃     min ┃     max ┃     p99 ┃     p50 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━┩
│        Request Latency (ms) │ 2456.89 │ 2012.45 │ 2890.34 │ 2890.34 │ 2398.12 │
│    Time to First Token (ms) │  478.90 │  398.23 │  567.89 │  567.89 │  467.34 │
│    Inter Token Latency (ms) │   19.45 │   15.67 │   24.12 │   24.12 │   19.01 │
│ Output Token Count (tokens) │  156.00 │  128.00 │  185.00 │  185.00 │  154.00 │
│  Request Throughput (req/s) │    2.34 │       - │       - │       - │       - │
└─────────────────────────────┴─────────┴─────────┴─────────┴─────────┴─────────┘

JSON Export: artifacts/Qwen_Qwen2-VL-2B-Instruct-chat-concurrency1/profile_export_aiperf.json
```
