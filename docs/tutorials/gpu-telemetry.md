<!--
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->

# GPU Telemetry with AIPerf

This guide shows you how to collect GPU metrics (power, utilization, memory, temperature, etc.) during AIPerf benchmarking. GPU telemetry provides insights into GPU performance and resource usage while running inference workloads.

## Overview

This guide covers two setup paths depending on your inference backend:

### Path 1: Dynamo (Built-in DCGM)
If you're using **Dynamo**, it comes with DCGM pre-configured on port 9401. No additional setup needed!

- ✅ No DCGM setup required
- ✅ Already on port 9401 (AIPerf's default)
- ✅ Just use `--gpu-telemetry` flag to enable console display

### Path 2: Custom DCGM (vLLM, SGLang, TRT-LLM, etc.)
If you're using **any other inference backend**, you'll need to set up DCGM separately.

- Works with: vLLM, SGLang, TRT-LLM, OpenAI-compatible servers, etc.
- Requires: DCGM Exporter container setup
- Configure: Forward to port 9401 (AIPerf's default)
- Result: Same `--gpu-telemetry` flag usage as Dynamo

## Prerequisites

- NVIDIA GPU with CUDA support
- Docker installed and configured
- Access to NVIDIA container registry (`nvcr.io`)
- AIPerf installed or access to AIPerf container

## Understanding GPU Telemetry in AIPerf

AIPerf provides GPU telemetry collection with the `--gpu-telemetry` flag. Here's how it works:

### How the `--gpu-telemetry` Flag Works

| Usage | Command | What Gets Collected (If Available) | Console Display | CSV/JSON Export |
|-------|---------|---------------------|-----------------|-----------------|
| **No flag** | `aiperf profile --model MODEL ...` | `http://localhost:9401/metrics` | ❌ No | ✅ Yes |
| **Flag only** | `aiperf profile --model MODEL ... --gpu-telemetry` | `http://localhost:9401/metrics` | ✅ Yes | ✅ Yes |
| **Custom URLs** | `aiperf profile --model MODEL ... --gpu-telemetry http://node1:9400/metrics http://node2:9400/metrics` | `http://localhost:9401/metrics` + custom URLs | ✅ Yes | ✅ Yes |

> [!IMPORTANT]
> The default endpoint `http://localhost:9401/metrics` is ALWAYS attempted for telemetry collection, regardless of whether the `--gpu-telemetry` flag is used. The flag primarily controls whether metrics are displayed on the console and allows you to specify additional custom DCGM exporter endpoints.

---

# Path 1: Using Dynamo

Dynamo includes DCGM out of the box on port 9401 - no extra setup needed!

## Setup Dynamo Server

<!-- setup-dynamo-gpu-telemetry-openai-endpoint-server -->
```bash
# Set environment variables
export AIPERF_REPO_TAG="main"
export DYNAMO_PREBUILT_IMAGE_TAG="nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.5.0"
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
```
<!-- /setup-dynamo-gpu-telemetry-openai-endpoint-server -->

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

## Verify Dynamo is Running

<!-- health-check-dynamo-gpu-telemetry-openai-endpoint-server -->
```bash
# Wait for Dynamo API to be ready (up to 15 minutes)
timeout 900 bash -c 'while [ "$(curl -s -o /dev/null -w "%{http_code}" localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"Qwen/Qwen3-0.6B\",\"messages\":[{\"role\":\"user\",\"content\":\"a\"}],\"max_completion_tokens\":1}")" != "200" ]; do sleep 2; done' || { echo "Dynamo not ready after 15min"; exit 1; }

# Wait for DCGM metrics to be available (up to 2 minutes after Dynamo is ready)
echo "Dynamo API ready, waiting for DCGM metrics to be available..."
timeout 120 bash -c 'while ! curl -s localhost:9401/metrics | grep -q "DCGM_FI_DEV_GPU_UTIL"; do echo "Waiting for DCGM metrics..."; sleep 5; done' || { echo "GPU utilization metrics not found in Dynamo DCGM after 2min"; exit 1; }

echo "DCGM GPU metrics are now available"
```
<!-- /health-check-dynamo-gpu-telemetry-openai-endpoint-server -->

## Run AIPerf Benchmark

<!-- aiperf-run-dynamo-gpu-telemetry-openai-endpoint-server -->
```bash
# Run AIPerf with Dynamo's GPU telemetry (port 9401)
# Note: Since Dynamo uses port 9401 (AIPerf's default), the --gpu-telemetry flag
# enables console display.
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --endpoint /v1/chat/completions \
    --streaming \
    --url localhost:8080 \
    --synthetic-input-tokens-mean 100 \
    --synthetic-input-tokens-stddev 0 \
    --output-tokens-mean 200 \
    --output-tokens-stddev 0 \
    --extra-inputs min_tokens:200 \
    --extra-inputs ignore_eos:true \
    --concurrency 4 \
    --request-count 64 \
    --warmup-request-count 1 \
    --conversation-num 8 \
    --random-seed 100 \
    --gpu-telemetry
```
<!-- /aiperf-run-dynamo-gpu-telemetry-openai-endpoint-server -->

---

# Path 2: Using Any Other Backend

This path works with **vLLM, SGLang, TRT-LLM, or any inference server**. We'll use vLLM as an example.

## Setup vLLM Server with DCGM

The setup includes three steps: creating a custom metrics configuration, starting the DCGM Exporter, and launching the vLLM server.

<!-- setup-vllm-gpu-telemetry-openai-endpoint-server -->
```bash
# ============================================================================
# Step 1: Create custom GPU metrics configuration file
# ============================================================================
cat > custom_gpu_metrics.csv << 'EOF'
# Format
# If line starts with a '#' it is considered a comment
# DCGM FIELD, Prometheus metric type, help message

# Clocks
DCGM_FI_DEV_SM_CLOCK, gauge, SM clock frequency (in MHz)
DCGM_FI_DEV_MEM_CLOCK, gauge, Memory clock frequency (in MHz)

# Temperature
DCGM_FI_DEV_MEMORY_TEMP, gauge, Memory temperature (in °C)
DCGM_FI_DEV_GPU_TEMP, gauge, GPU temperature (in °C)

# Power
DCGM_FI_DEV_POWER_USAGE, gauge, Power draw (in W)
DCGM_FI_DEV_POWER_MGMT_LIMIT, gauge, Power management limit (in W)
DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION, counter, Total energy consumption since boot (in mJ)

# Memory usage
DCGM_FI_DEV_FB_FREE, gauge, Framebuffer memory free (in MiB)
DCGM_FI_DEV_FB_TOTAL, gauge, Total framebuffer memory (in MiB)
DCGM_FI_DEV_FB_USED, gauge, Framebuffer memory used (in MiB)

# Utilization
DCGM_FI_DEV_GPU_UTIL, gauge, GPU utilization (in %)
DCGM_FI_DEV_MEM_COPY_UTIL, gauge, Memory copy utilization (in %)

# Errors and Violations
DCGM_FI_DEV_XID_ERRORS, gauge, Value of the last XID error encountered
DCGM_FI_DEV_POWER_VIOLATION, counter, Throttling duration due to power constraints (in us)
DCGM_FI_DEV_THERMAL_VIOLATION, counter, Throttling duration due to thermal constraints (in us)
EOF

# ============================================================================
# Step 2: Start DCGM Exporter
# ============================================================================
# Set environment variables
export DCGM_EXPORTER_IMAGE="nvcr.io/nvidia/k8s/dcgm-exporter:4.2.0-4.1.0-ubuntu22.04"

# Clean up any existing dcgm-exporter container
docker stop dcgm-exporter 2>/dev/null || true
docker rm dcgm-exporter 2>/dev/null || true

# Start DCGM Exporter container (forwards port 9400 → 9401)
docker run -d --name dcgm-exporter \
  --gpus all \
  --cap-add SYS_ADMIN \
  -p 9401:9400 \
  -v "$PWD/custom_gpu_metrics.csv:/etc/dcgm-exporter/custom.csv" \
  -e DCGM_EXPORTER_INTERVAL=33 \
  ${DCGM_EXPORTER_IMAGE} \
  -f /etc/dcgm-exporter/custom.csv

# Wait for DCGM to start
sleep 10

# ============================================================================
# Step 3: Start vLLM Inference Server
# ============================================================================
export MODEL="Qwen/Qwen3-0.6B"
export AIPERF_REPO_TAG="main"

# Pull the vLLM image
docker pull vllm/vllm-openai:latest

# Start vLLM server
docker run -d --name vllm-server \
  --gpus all \
  -p 8000:8000 \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen3-0.6B \
  --host 0.0.0.0 \
  --port 8000
```
<!-- /setup-vllm-gpu-telemetry-openai-endpoint-server -->

> [!TIP]
> You can customize the `custom_gpu_metrics.csv` file by commenting out metrics you don't need. Lines starting with `#` are ignored.

**Key Configuration:**
- `-p 9401:9400` - Forward container's port 9400 to host's port 9401 (AIPerf's default)
- `-e DCGM_EXPORTER_INTERVAL=33` - Collect metrics every 33ms for fine-grained profiling
- `-v custom_gpu_metrics.csv:...` - Mount your custom metrics configuration

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

> [!NOTE]
> Replace the vLLM command above with your preferred backend (SGLang, TRT-LLM, etc.). The DCGM setup works with any server.

## Verify Everything is Running

<!-- health-check-vllm-gpu-telemetry-openai-endpoint-server -->
```bash
# Wait for vLLM inference server to be ready (up to 15 minutes)
timeout 900 bash -c 'while [ "$(curl -s -o /dev/null -w "%{http_code}" localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"Qwen/Qwen3-0.6B\",\"messages\":[{\"role\":\"user\",\"content\":\"test\"}],\"max_tokens\":1}")" != "200" ]; do sleep 2; done' || { echo "vLLM not ready after 15min"; exit 1; }

# Wait for DCGM Exporter metrics to be available (up to 2 minutes after vLLM is ready)
echo "vLLM ready, waiting for DCGM metrics to be available..."
timeout 120 bash -c 'while ! curl -s localhost:9401/metrics | grep -q "DCGM_FI_DEV_GPU_UTIL"; do echo "Waiting for DCGM metrics..."; sleep 5; done' || { echo "GPU utilization metrics not found after 2min"; exit 1; }

echo "DCGM GPU metrics are now available"
```
<!-- /health-check-vllm-gpu-telemetry-openai-endpoint-server -->

## Run AIPerf Benchmark

<!-- aiperf-run-vllm-gpu-telemetry-openai-endpoint-server -->
```bash
# Run AIPerf with custom DCGM GPU telemetry
# Note: DCGM is forwarded to port 9401 (AIPerf's default), so --gpu-telemetry
# flag enables console display without needing to specify a URL
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --endpoint /v1/chat/completions \
    --streaming \
    --url localhost:8080 \
    --synthetic-input-tokens-mean 100 \
    --synthetic-input-tokens-stddev 0 \
    --output-tokens-mean 200 \
    --output-tokens-stddev 0 \
    --extra-inputs min_tokens:200 \
    --extra-inputs ignore_eos:true \
    --concurrency 4 \
    --request-count 64 \
    --warmup-request-count 1 \
    --conversation-num 8 \
    --random-seed 100 \
    --gpu-telemetry
```
<!-- /aiperf-run-vllm-gpu-telemetry-openai-endpoint-server -->

## Example output:

```
                                  NVIDIA AIPerf | GPU Telemetry Summary
                                      1/1 DCGM endpoints reachable
                                           • localhost:9401 ✔

                                localhost:9401 | GPU 0 | NVIDIA H100 PCIe
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━┓
┃                       Metric ┃      avg ┃      min ┃      max ┃      p99 ┃      p90 ┃      p50 ┃  std ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━┩
│          GPU Power Usage (W) │    79.58 │    79.58 │    79.58 │    79.58 │    79.58 │    79.58 │ 0.00 │
│      Energy Consumption (MJ) │     0.04 │     0.04 │     0.04 │     0.04 │     0.04 │     0.04 │ 0.00 │
│          GPU Utilization (%) │     0.00 │     0.00 │     0.00 │     0.00 │     0.00 │     0.00 │ 0.00 │
│  Memory Copy Utilization (%) │     0.00 │     0.00 │     0.00 │     0.00 │     0.00 │     0.00 │ 0.00 │
│         GPU Memory Used (GB) │    77.23 │    77.23 │    77.23 │    77.23 │    77.23 │    77.23 │ 0.00 │
│     SM Clock Frequency (MHz) │ 1,755.00 │ 1,755.00 │ 1,755.00 │ 1,755.00 │ 1,755.00 │ 1,755.00 │ 0.00 │
│ Memory Clock Frequency (MHz) │ 1,593.00 │ 1,593.00 │ 1,593.00 │ 1,593.00 │ 1,593.00 │ 1,593.00 │ 0.00 │
│      Memory Temperature (°C) │    51.00 │    51.00 │    51.00 │    51.00 │    51.00 │    51.00 │ 0.00 │
│         GPU Temperature (°C) │    36.00 │    36.00 │    36.00 │    36.00 │    36.00 │    36.00 │ 0.00 │
│           XID Errors (count) │     0.00 │     0.00 │     0.00 │     0.00 │     0.00 │     0.00 │ 0.00 │
└──────────────────────────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────┘

                                localhost:9401 | GPU 1 | NVIDIA H100 PCIe
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━┓
┃                       Metric ┃      avg ┃      min ┃      max ┃      p99 ┃      p90 ┃      p50 ┃  std ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━┩
│          GPU Power Usage (W) │    46.63 │    46.63 │    46.63 │    46.63 │    46.63 │    46.63 │ 0.00 │
│      Energy Consumption (MJ) │     0.04 │     0.04 │     0.04 │     0.04 │     0.04 │     0.04 │ 0.00 │
│          GPU Utilization (%) │     0.00 │     0.00 │     0.00 │     0.00 │     0.00 │     0.00 │ 0.00 │
│  Memory Copy Utilization (%) │     0.00 │     0.00 │     0.00 │     0.00 │     0.00 │     0.00 │ 0.00 │
│         GPU Memory Used (GB) │     0.00 │     0.00 │     0.00 │     0.00 │     0.00 │     0.00 │ 0.00 │
│     SM Clock Frequency (MHz) │   345.00 │   345.00 │   345.00 │   345.00 │   345.00 │   345.00 │ 0.00 │
│ Memory Clock Frequency (MHz) │ 1,593.00 │ 1,593.00 │ 1,593.00 │ 1,593.00 │ 1,593.00 │ 1,593.00 │ 0.00 │
│      Memory Temperature (°C) │    45.00 │    45.00 │    45.00 │    45.00 │    45.00 │    45.00 │ 0.00 │
│         GPU Temperature (°C) │    28.00 │    28.00 │    28.00 │    28.00 │    28.00 │    28.00 │ 0.00 │
│           XID Errors (count) │     0.00 │     0.00 │     0.00 │     0.00 │     0.00 │     0.00 │ 0.00 │
└──────────────────────────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────┘

                                localhost:9401 | GPU 2 | NVIDIA H100 PCIe
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━┓
┃                       Metric ┃      avg ┃      min ┃      max ┃      p99 ┃      p90 ┃      p50 ┃  std ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━┩
│          GPU Power Usage (W) │    45.66 │    45.66 │    45.66 │    45.66 │    45.66 │    45.66 │ 0.00 │
│      Energy Consumption (MJ) │     0.04 │     0.04 │     0.04 │     0.04 │     0.04 │     0.04 │ 0.00 │
│          GPU Utilization (%) │     0.00 │     0.00 │     0.00 │     0.00 │     0.00 │     0.00 │ 0.00 │
│  Memory Copy Utilization (%) │     0.00 │     0.00 │     0.00 │     0.00 │     0.00 │     0.00 │ 0.00 │
│         GPU Memory Used (GB) │     0.00 │     0.00 │     0.00 │     0.00 │     0.00 │     0.00 │ 0.00 │
│     SM Clock Frequency (MHz) │   345.00 │   345.00 │   345.00 │   345.00 │   345.00 │   345.00 │ 0.00 │
│ Memory Clock Frequency (MHz) │ 1,593.00 │ 1,593.00 │ 1,593.00 │ 1,593.00 │ 1,593.00 │ 1,593.00 │ 0.00 │
│      Memory Temperature (°C) │    47.00 │    47.00 │    47.00 │    47.00 │    47.00 │    47.00 │ 0.00 │
│         GPU Temperature (°C) │    26.00 │    26.00 │    26.00 │    26.00 │    26.00 │    26.00 │ 0.00 │
│           XID Errors (count) │     0.00 │     0.00 │     0.00 │     0.00 │     0.00 │     0.00 │ 0.00 │
└──────────────────────────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────┘
```

## Multi-Node GPU Telemetry Example

For distributed setups with multiple nodes, you can collect GPU telemetry from all nodes simultaneously:

```bash
# Example: Collecting telemetry from 3 nodes in a distributed setup
# Note: The default http://localhost:9401/metrics is always attempted in addition to these URLs
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --endpoint /v1/chat/completions \
    --streaming \
    --url localhost:8080 \
    --synthetic-input-tokens-mean 100 \
    --synthetic-input-tokens-stddev 0 \
    --output-tokens-mean 200 \
    --output-tokens-stddev 0 \
    --extra-inputs min_tokens:200 \
    --extra-inputs ignore_eos:true \
    --concurrency 4 \
    --request-count 64 \
    --warmup-request-count 1 \
    --conversation-num 8 \
    --random-seed 100 \
    --gpu-telemetry http://node1:9400/metrics http://node2:9400/metrics http://node3:9400/metrics
```

This will collect GPU metrics from:
- `http://localhost:9401/metrics` (default, always attempted)
- `http://node1:9400/metrics` (custom node 1)
- `http://node2:9400/metrics` (custom node 2)

All metrics are displayed on the console and saved to the output CSV and JSON files, with GPU indices and hostnames distinguishing metrics from different nodes.