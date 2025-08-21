<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Profiling using AIPerf

This tutorial will demonstrate how you can use AIPerf to measure the performance of
models using various inference solutions.

### Table of Contents
- [Profile GPT-OSS-20B using Dynamo](#dynamo-gpt-oss-20B)
- [Profile Qwen3-0.6B using vllm](#vllm-qwen3-0.6B)

</br>

## Profile GPT-OSS-20B using Dynamo <a id="dynamo-gpt-oss-20B">

[!Note] The most up to date installation instructions for Dynamo are available on [Github](https://github.com/ai-dynamo/dynamo?tab=readme-ov-file#1-initial-setup)

```bash
# set environment variables
export AIPERF_REPO_TAG="main"
export DYNAMO_PREBUILT_IMAGE_TAG="nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.4.0"
export MODEL="openai/gpt-oss-20b"

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

# At this point, Dynamo may not be ready.
# The following command will return when Dynamo is ready for requests.
while [ "$(curl -s -o /dev/null -w '%{http_code}' localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"'"${MODEL}"'","messages":[{"role":"user","content":"a"}],"max_completion_tokens":1}')" != "200" ]; do sleep 1; done

# Profile the model
aiperf profile \
    --model openai/gpt-oss-20b \
    --endpoint-type chat \
    --endpoint /v1/chat/completions \
    --streaming \
    --url localhost:8080 \
    --synthetic-input-tokens-mean 1000 \
    --synthetic-input-tokens-stddev 0 \
    --output-tokens-mean 2000 \
    --output-tokens-stddev 0 \
    --extra-inputs max_tokens:2000 \
    --extra-inputs min_tokens:2000 \
    --extra-inputs ignore_eos:true \
    --extra-inputs "{\"nvext\":{\"ignore_eos\":true}}" \
    --concurrency 2048 \
    --request-count 6144 \
    --warmup-request-count 1000 \
    --conversation-num 8000 \
    --random-seed 100 \
    -v \
    -H 'Authorization: Bearer NOT USED' \
    -H 'Accept: text/event-stream'
```

## Profile Qwen3-0.6B using vllm <a id="vllm-qwen3-0.6B">
```bash
# Install vLLM from pip:
pip install vllm

# Load and run the model:
vllm serve "Qwen/Qwen3-0.6B"

uv venv
source .venv/bin/activate
pip install git+https://github.com/ai-dynamo/aiperf.git

aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --endpoint /v1/chat/completions \
    --streaming \
    --request-rate 1000 \
    --request-count 6500
```
