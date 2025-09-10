<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Profiling with AIPerf

This tutorial shows how to measure model performance across different inference solutions using AIPerf.

### Table of Contents
- [Profile Qwen3-0.6B Using Dynamo](#dynamo-qwen3-0.6B)
- [Profile Qwen3-0.6B Using vllm](#vllm-qwen3-0.6B)

## Profile Qwen3-0.6B Using Dynamo <a id="dynamo-qwen3-0.6B">

> [!NOTE]
> The latest installation instructions for Dynamo are available on [Github](https://github.com/ai-dynamo/dynamo?tab=readme-ov-file#1-initial-setup)

```bash
# Set environment variables
export AIPERF_REPO_TAG="main"
export DYNAMO_PREBUILT_IMAGE_TAG="nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.4.0"
export MODEL="Qwen/Qwen3-0.6B"

# Download the Dyanmo container
docker pull ${DYNAMO_PREBUILT_IMAGE_TAG}

export DYNAMO_REPO_TAG=$(docker run --rm --entrypoint "" ${DYNAMO_PREBUILT_IMAGE_TAG} cat /workspace/version.txt | cut -d'+' -f2)


# Start up required services
curl -O https://raw.githubusercontent.com/ai-dynamo/dynamo/${DYNAMO_REPO_TAG}/deploy/docker-compose.yml
docker compose -f docker-compose.yml down || true
docker compose -f docker-compose.yml up -d

# Launch Dynamo in the background
docker run \
  --rm \
  --gpus all \
  --network host \
  ${DYNAMO_PREBUILT_IMAGE_TAG} \
    /bin/bash -c "python3 -m dynamo.frontend & python3 -m dynamo.vllm --model ${MODEL} --enforce-eager --no-enable-prefix-caching" > server.log 2>&1 &

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

# It can take some time for Dynamo to become ready.
# The following command returns when Dynamo is ready to accept requests.
while [ "$(curl -s -o /dev/null -w '%{http_code}' localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"'"${MODEL}"'","messages":[{"role":"user","content":"a"}],"max_completion_tokens":1}')" != "200" ]; do sleep 1; done

# Profile the model
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --endpoint /v1/chat/completions \
    --streaming \
    --url localhost:8000 \
    --synthetic-input-tokens-mean 1000 \
    --synthetic-input-tokens-stddev 0 \
    --output-tokens-mean 2000 \
    --output-tokens-stddev 0 \
    --extra-inputs min_tokens:2000 \
    --extra-inputs ignore_eos:true \
    --concurrency 2048 \
    --request-count 6144 \
    --warmup-request-count 1000 \
    --conversation-num 8000 \
    --random-seed 100 \
    -v \
    -H 'Authorization: Bearer NOT USED' \
    -H 'Accept: text/event-stream'
```

## Profile Qwen3-0.6B Using vLLM <a id="vllm-qwen3-0.6B">
```bash
# Install vLLM from pip:
pip install vllm

# Load and run the model:
vllm serve "Qwen/Qwen3-0.6B"

uv venv
source .venv/bin/activate
uv pip install git+https://github.com/ai-dynamo/aiperf.git

aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --endpoint /v1/chat/completions \
    --streaming \
    --request-rate 1000 \
    --request-count 6500
```

## Profile Qwen3-0.6B Using vLLM and Docker <a id="vllm-qwen3-0.6B-docker">


```bash
# Install the latest vLLM docker container:
docker run \
  -it \
  --rm \
  --gpus all \
  --network host \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen3-0.6B

# In a separate terminal, ensure dependencies are installed
apt update && apt install -y curl git
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.10
source .venv/bin/activate

# Install and Run AIPerf
uv pip install git+https://github.com/ai-dynamo/aiperf.git

aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --endpoint /v1/chat/completions \
    --streaming \
    --request-rate 100 \
    --request-count 650
```