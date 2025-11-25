<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Profiling with AIPerf

This tutorial will demonstrate how you can use AIPerf to measure the performance of
models using various inference solutions.

### Table of Contents
- [Profile Qwen3-0.6B Using Dynamo](#dynamo-qwen3-0.6B)
- [Profile Qwen3-0.6B Using vllm](#vllm-qwen3-0.6B)

## Profile Qwen3-0.6B Using Dynamo <a id="dynamo-qwen3-0.6B">

> [!NOTE]
> The latest installation instructions for Dynamo are available on [Github](https://github.com/ai-dynamo/dynamo?tab=readme-ov-file#1-initial-setup)

<!-- setup-dynamo-default-openai-endpoint-server -->
```bash
# Set environment variables
export AIPERF_REPO_TAG="main"
export DYNAMO_PREBUILT_IMAGE_TAG="nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.6.1"
export MODEL="Qwen/Qwen3-0.6B"

# Download the Dynamo container
docker pull ${DYNAMO_PREBUILT_IMAGE_TAG}

# Launch Dynamo with nats-server and etcd in the same container
docker run \
  --rm \
  --gpus all \
  --network host \
  ${DYNAMO_PREBUILT_IMAGE_TAG} \
    /bin/bash -c "nats-server -js & while ! timeout 1 bash -c 'cat < /dev/null > /dev/tcp/localhost/4222' 2>/dev/null; do sleep 0.1; done && etcd --listen-client-urls http://0.0.0.0:2379 --advertise-client-urls http://0.0.0.0:2379 --data-dir /tmp/etcd & while ! timeout 1 bash -c 'cat < /dev/null > /dev/tcp/localhost/2379' 2>/dev/null; do sleep 0.1; done && python3 -m dynamo.frontend & python3 -m dynamo.vllm --model ${MODEL} --enforce-eager --no-enable-prefix-caching" > server.log 2>&1 &
```
<!-- /setup-dynamo-default-openai-endpoint-server -->

```bash
# Set up AIPerf
docker run \
  -it \
  --rm \
  --gpus all \
  --network host \
  -e AIPERF_REPO_TAG=${AIPERF_REPO_TAG} \
  -e MODEL=${MODEL} \
  ubuntu:24.04

apt update && apt install -y curl git

curl -LsSf https://astral.sh/uv/install.sh | sh

source $HOME/.local/bin/env

uv venv --python 3.10

source .venv/bin/activate

git clone -b ${AIPERF_REPO_TAG} --depth 1 https://github.com/ai-dynamo/aiperf.git

uv pip install ./aiperf
```
<!-- health-check-dynamo-default-openai-endpoint-server -->
```bash
timeout 900 bash -c 'while [ "$(curl -s -o /dev/null -w "%{http_code}" localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"Qwen/Qwen3-0.6B\",\"messages\":[{\"role\":\"user\",\"content\":\"a\"}],\"max_completion_tokens\":1}")" != "200" ]; do sleep 2; done' || { echo "Dynamo not ready after 15min"; exit 1; }
```
<!-- /health-check-dynamo-default-openai-endpoint-server -->
<!-- aiperf-run-dynamo-default-openai-endpoint-server -->
```bash
# Profile the model
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --endpoint /v1/chat/completions \
    --streaming \
    --url localhost:8000 \
    --synthetic-input-tokens-mean 100 \
    --synthetic-input-tokens-stddev 0 \
    --output-tokens-mean 200 \
    --output-tokens-stddev 0 \
    --extra-inputs min_tokens:200 \
    --extra-inputs ignore_eos:true \
    --concurrency 4 \
    --request-count 64 \
    --warmup-request-count 1 \
    --num-dataset-entries 8 \
    --random-seed 100
```

<!-- /aiperf-run-dynamo-default-openai-endpoint-server -->

## Profile Qwen3-0.6B using vllm <a id="vllm-qwen3-0.6B">
<!-- setup-vllm-default-openai-endpoint-server -->
```bash
# Pull and run vLLM Docker container:
docker pull vllm/vllm-openai:latest
docker run --gpus all -p 8000:8000 vllm/vllm-openai:latest \
  --model Qwen/Qwen3-0.6B \
  --reasoning-parser qwen3 \
  --host 0.0.0.0 --port 8000
```
<!-- /setup-vllm-default-openai-endpoint-server -->

<!-- health-check-vllm-default-openai-endpoint-server -->
```bash
timeout 900 bash -c 'while [ "$(curl -s -o /dev/null -w "%{http_code}" localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"Qwen/Qwen3-0.6B\",\"messages\":[{\"role\":\"user\",\"content\":\"test\"}],\"max_tokens\":1}")" != "200" ]; do sleep 2; done' || { echo "vLLM not ready after 15min"; exit 1; }
```
<!-- /health-check-vllm-default-openai-endpoint-server -->


<!-- aiperf-run-vllm-default-openai-endpoint-server -->
```bash
# Profile the model
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --endpoint /v1/chat/completions \
    --streaming \
    --request-rate 32 \
    --request-count 64 \
    --url localhost:8000
```
<!-- /aiperf-run-vllm-default-openai-endpoint-server -->