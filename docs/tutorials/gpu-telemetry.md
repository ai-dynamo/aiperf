<!--
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->

# GPU Telemetry with AIPerf

This guide shows you how to collect GPU metrics (power, utilization, memory, temperature, etc.) during AIPerf benchmarking. GPU telemetry provides insights into GPU performance and resource usage while running inference workloads.

## Overview

This guide covers two setup paths depending on your inference backend:

### Path 1: Dynamo (Built-in DCGM)
If you're using **Dynamo**, it comes with DCGM pre-configured on port 9401. No additional setup needed! Just use the `--gpu-telemetry` flag to enable console display and optionally add additional DCGM url endpoints. URLs can be specified with or without the `http://` prefix (e.g., `localhost:9400` or `http://localhost:9400`).

### Path 2: Other Inference Servers (Custom DCGM)
If you're using **any other inference backend**, you'll need to set up DCGM separately.

## Prerequisites

- NVIDIA GPU with CUDA support
- Docker installed and configured

## Understanding GPU Telemetry in AIPerf

AIPerf provides GPU telemetry collection with the `--gpu-telemetry` flag. Here's how it works:

### How the `--gpu-telemetry` Flag Works

| Usage | Command | What Gets Collected (If Available) | Console Display | Dashboard View | CSV/JSON Export |
|-------|---------|---------------------|-----------------|----------------|-----------------|
| **No flag** | `aiperf profile --model MODEL ...` | `http://localhost:9400/metrics` + `http://localhost:9401/metrics` | ❌ No | ❌ No | ✅ Yes |
| **Flag only** | `aiperf profile --model MODEL ... --gpu-telemetry` | `http://localhost:9400/metrics` + `http://localhost:9401/metrics` | ✅ Yes | ❌ No | ✅ Yes |
| **Dashboard mode** | `aiperf profile --model MODEL ... --gpu-telemetry dashboard` | `http://localhost:9400/metrics` + `http://localhost:9401/metrics` | ✅ Yes | ✅ Yes ([see dashboard](#real-time-dashboard-view)) | ✅ Yes |
| **Custom URLs** | `aiperf profile --model MODEL ... --gpu-telemetry node1:9400 http://node2:9400/metrics` | `http://localhost:9400/metrics` + `http://localhost:9401/metrics` + [custom URLs](#multi-node-gpu-telemetry-example) | ✅ Yes | ❌ No | ✅ Yes |
| **Dashboard + URLs** | `aiperf profile --model MODEL ... --gpu-telemetry dashboard localhost:9400` | `http://localhost:9400/metrics` + `http://localhost:9401/metrics` + [custom URLs](#multi-node-gpu-telemetry-example) | ✅ Yes | ✅ Yes ([see dashboard](#real-time-dashboard-view)) | ✅ Yes |
| **Custom metrics** | `aiperf profile --model MODEL ... --gpu-telemetry custom_gpu_metrics.csv` | `http://localhost:9400/metrics` + `http://localhost:9401/metrics` + [custom metrics from CSV](#customizing-displayed-metrics) | ✅ Yes | ❌ No | ✅ Yes |
| **Disabled** | `aiperf profile --model MODEL ... --no-gpu-telemetry` | None | ❌ No | ❌ No | ❌ No |

> [!IMPORTANT]
> The default endpoints `http://localhost:9400/metrics` and `http://localhost:9401/metrics` are ALWAYS attempted for telemetry collection, regardless of whether the `--gpu-telemetry` flag is used. The flag primarily controls whether metrics are displayed on the console and allows you to specify additional custom DCGM exporter endpoints. To completely disable GPU telemetry collection, use `--no-gpu-telemetry`.

> [!NOTE]
> When specifying custom DCGM exporter URLs, the `http://` prefix is optional. URLs like `localhost:9400` will automatically be treated as `http://localhost:9400`. Both formats work identically.

### Real-Time Dashboard View

Adding `dashboard` to the `--gpu-telemetry` flag enables a live terminal UI (TUI) that displays GPU metrics in real-time during your benchmark runs:

```bash
aiperf profile --model MODEL ... --gpu-telemetry dashboard
```

---

# 1: Using Dynamo

Dynamo includes DCGM out of the box on port 9401 - no extra setup needed!

## Setup Dynamo Server

```bash
# Set environment variables
export AIPERF_REPO_TAG="main"
export DYNAMO_PREBUILT_IMAGE_TAG="nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.6.1"
export MODEL="Qwen/Qwen3-0.6B"

# Download the Dynamo container
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

```bash
# Wait for Dynamo API to be ready (up to 15 minutes)
timeout 900 bash -c 'while [ "$(curl -s -o /dev/null -w "%{http_code}" localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"Qwen/Qwen3-0.6B\",\"messages\":[{\"role\":\"user\",\"content\":\"a\"}],\"max_completion_tokens\":1}")" != "200" ]; do sleep 2; done' || { echo "Dynamo not ready after 15min"; exit 1; }
```
```bash
# Wait for DCGM Exporter to be ready (up to 2 minutes after Dynamo is ready)
echo "Dynamo ready, waiting for DCGM metrics to be available..."
timeout 120 bash -c 'while true; do STATUS=$(curl -s -o /dev/null -w "%{http_code}" localhost:9401/metrics); if [ "$STATUS" = "200" ]; then if curl -s localhost:9401/metrics | grep -q "DCGM_FI_DEV_GPU_UTIL"; then break; fi; fi; echo "Waiting for DCGM metrics..."; sleep 5; done' || { echo "GPU utilization metrics not found after 2min"; exit 1; }
echo "DCGM GPU metrics are now available"
```

## Run AIPerf Benchmark

```bash
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
    --random-seed 100 \
    --gpu-telemetry
```

**Sample Output (Successful Run):**
```
INFO     Starting AIPerf System
INFO     AIPerf System is PROFILING

Profiling: 64/64 |████████████████████████| 100% [00:45<00:00]

INFO     Benchmark completed successfully


                          NVIDIA AIPerf | GPU Telemetry Summary
                               1/1 DCGM endpoints reachable
                                    • localhost:9401 ✔

                      localhost:9401 | GPU 0 | NVIDIA H100 80GB HBM3
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━┓
┃                       Metric ┃      avg ┃      min ┃      max ┃      p99 ┃      p90 ┃      p50 ┃   std ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━┩
│          GPU Power Usage (W) │   348.69 │   120.57 │   386.02 │   386.02 │   386.02 │   378.34 │ 85.97 │
│      Energy Consumption (MJ) │     0.24 │     0.23 │     0.25 │     0.25 │     0.25 │     0.23 │  0.01 │
│          GPU Utilization (%) │    45.82 │     0.00 │    66.00 │    66.00 │    66.00 │    66.00 │ 24.52 │
│  Memory Copy Utilization (%) │    21.10 │     0.00 │    29.00 │    29.00 │    29.00 │    29.00 │ 10.11 │
│         GPU Memory Used (GB) │    92.70 │    92.70 │    92.70 │    92.70 │    92.70 │    92.70 │  0.00 │
│         GPU Memory Free (GB) │     9.39 │     9.39 │     9.39 │     9.39 │     9.39 │     9.39 │  0.00 │
│     SM Clock Frequency (MHz) │ 1,980.00 │ 1,980.00 │ 1,980.00 │ 1,980.00 │ 1,980.00 │ 1,980.00 │  0.00 │
│ Memory Clock Frequency (MHz) │ 2,619.00 │ 2,619.00 │ 2,619.00 │ 2,619.00 │ 2,619.00 │ 2,619.00 │  0.00 │
│      Memory Temperature (°C) │    45.99 │    41.00 │    48.00 │    48.00 │    48.00 │    46.00 │  2.08 │
│         GPU Temperature (°C) │    38.87 │    33.00 │    41.00 │    41.00 │    41.00 │    39.00 │  2.38 │
│           XID Errors (count) │     0.00 │     0.00 │     0.00 │     0.00 │     0.00 │     0.00 │  0.00 │
└──────────────────────────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴───────┘

CLI Command: aiperf profile --model "Qwen/Qwen3-0.6B" --endpoint-type "chat" ...
JSON Export: artifacts/Qwen_Qwen3-0.6B-chat-concurrency4/profile_export_aiperf.json
GPU Telemetry: artifacts/Qwen_Qwen3-0.6B-chat-concurrency4/gpu_telemetry_export.json
```

The GPU telemetry table displays real-time metrics collected from DCGM during the benchmark. Each GPU is shown with its metrics aggregated across the benchmark duration.

> [!TIP]
> The `dashboard` keyword enables a live terminal UI for real-time GPU telemetry visualization. Press `5` to maximize the GPU Telemetry panel during the benchmark run.

---

# 2: Using Other Inference Server

This path works with **vLLM, SGLang, TRT-LLM, or any inference server**. We'll use vLLM as an example.

## Setup vLLM Server with DCGM

The setup includes three steps: creating a custom metrics configuration, starting the DCGM Exporter, and launching the vLLM server.

```bash
# Step 1: Create a custom metrics configuration
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

# Step 2: Start DCGM Exporter container (forwards port 9400 → 9401)
export DCGM_EXPORTER_IMAGE="nvcr.io/nvidia/k8s/dcgm-exporter:4.2.0-4.1.0-ubuntu22.04"

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

# Step 3: Start vLLM Inference Server
export MODEL="Qwen/Qwen3-0.6B"

docker pull vllm/vllm-openai:latest

docker run -d --name vllm-server \
  --gpus all \
  -p 8000:8000 \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen3-0.6B \
  --host 0.0.0.0 \
  --port 8000
```

> [!TIP]
> You can customize the `custom_gpu_metrics.csv` file by commenting out metrics you don't need. Lines starting with `#` are ignored.

**Key Configuration:**
- `-p 9401:9400` - Forward container's port 9400 to host's port 9401 (AIPerf's default)
- `-e DCGM_EXPORTER_INTERVAL=33` - Collect metrics every 33ms for fine-grained profiling
- `-v custom_gpu_metrics.csv:...` - Mount your custom metrics configuration

```bash
# Set up AIPerf
export AIPERF_REPO_TAG="main"

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

```bash
# Wait for vLLM inference server to be ready (up to 15 minutes)
timeout 900 bash -c 'while [ "$(curl -s -o /dev/null -w "%{http_code}" localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"Qwen/Qwen3-0.6B\",\"messages\":[{\"role\":\"user\",\"content\":\"test\"}],\"max_tokens\":1}")" != "200" ]; do sleep 2; done' || { echo "vLLM not ready after 15min"; exit 1; }

# Wait for DCGM Exporter metrics to be available (up to 2 minutes after vLLM is ready)
echo "vLLM ready, waiting for DCGM metrics to be available..."
timeout 120 bash -c 'while true; do OUTPUT=$(curl -s localhost:9401/metrics); if echo "$OUTPUT" | grep -q "DCGM_FI_DEV_GPU_UTIL"; then break; fi; echo "Waiting for DCGM metrics..."; sleep 5; done' || { echo "GPU utilization metrics not found after 2min"; exit 1; }
echo "DCGM GPU metrics are now available"
```

## Run AIPerf Benchmark

```bash
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
    --random-seed 100 \
    --gpu-telemetry
```

> [!TIP]
> The `dashboard` keyword enables a live terminal UI for real-time GPU telemetry visualization. Press `5` to maximize the GPU Telemetry panel during the benchmark run.

> [!TIP]
> The `dashboard` keyword enables a live terminal UI for real-time GPU telemetry visualization. Press `5` to maximize the GPU Telemetry panel during the benchmark run.

## Multi-Node GPU Telemetry Example

For distributed setups with multiple nodes, you can collect GPU telemetry from all nodes simultaneously:

```bash
# Example: Collecting telemetry from 3 nodes in a distributed setup
# Note: The default endpoints http://localhost:9400/metrics and http://localhost:9401/metrics
#       are always attempted in addition to these custom URLs
# URLs can be specified with or without the http:// prefix
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
    --random-seed 100 \
    --gpu-telemetry node1:9400 node2:9400 http://node3:9400/metrics
```

This will collect GPU metrics from:
- `http://localhost:9400/metrics` (default, always attempted)
- `http://localhost:9401/metrics` (default, always attempted)
- `http://node1:9400` (custom node 1, normalized from `node1:9400`)
- `http://node2:9400` (custom node 2, normalized from `node2:9400`)
- `http://node3:9400/metrics` (custom node 3)

All metrics are displayed on the console and saved to the output CSV and JSON files, with GPU indices and hostnames distinguishing metrics from different nodes.

## Customizing Displayed Metrics

You can customize which GPU metrics are displayed in AIPerf by creating a custom metrics CSV file and passing it to `--gpu-telemetry`:

```bash
aiperf profile --model MODEL ... --gpu-telemetry custom_gpu_metrics.csv

aiperf profile --model MODEL ... --gpu-telemetry localhost:9400 dashboard custom_gpu_metrics.csv
```

### Custom Metrics CSV Format

The CSV format is identical to DCGM exporter configuration. See the **vLLM setup section above** (Step 1: Create a custom metrics configuration) for the complete CSV format example with all available DCGM fields.

**Behavior**: Custom metrics **extend** (not replace) the 7 core default metrics:
- GPU Power Usage
- Energy Consumption
- GPU Utilization
- GPU Memory Used
- GPU Temperature
- XID Errors
- Power Violation

> [!NOTE]
> The file path can be absolute or relative. Use `.csv` extension so AIPerf can distinguish it from DCGM endpoint URLs.

> [!TIP]
> You can start with the example CSV from the vLLM setup section and customize it by commenting out metrics you don't need or adding new DCGM metrics.

## Example Console Display:

```
                                  NVIDIA AIPerf | GPU Telemetry Summary
                                       1/1 DCGM endpoints reachable
                                            • localhost:9401 ✔

                              localhost:9401 | GPU 0 | NVIDIA H100 80GB HBM3
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━┓
┃                       Metric ┃      avg ┃      min ┃      max ┃      p99 ┃      p90 ┃      p50 ┃   std ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━┩
│          GPU Power Usage (W) │   348.69 │   120.57 │   386.02 │   386.02 │   386.02 │   378.34 │ 85.97 │
│      Energy Consumption (MJ) │     0.24 │     0.23 │     0.25 │     0.25 │     0.25 │     0.23 │  0.01 │
│          GPU Utilization (%) │    45.82 │     0.00 │    66.00 │    66.00 │    66.00 │    66.00 │ 24.52 │
│  Memory Copy Utilization (%) │    21.10 │     0.00 │    29.00 │    29.00 │    29.00 │    29.00 │ 10.11 │
│         GPU Memory Used (GB) │    92.70 │    92.70 │    92.70 │    92.70 │    92.70 │    92.70 │  0.00 │
│         GPU Memory Free (GB) │     9.39 │     9.39 │     9.39 │     9.39 │     9.39 │     9.39 │  0.00 │
│     SM Clock Frequency (MHz) │ 1,980.00 │ 1,980.00 │ 1,980.00 │ 1,980.00 │ 1,980.00 │ 1,980.00 │  0.00 │
│ Memory Clock Frequency (MHz) │ 2,619.00 │ 2,619.00 │ 2,619.00 │ 2,619.00 │ 2,619.00 │ 2,619.00 │  0.00 │
│      Memory Temperature (°C) │    45.99 │    41.00 │    48.00 │    48.00 │    48.00 │    46.00 │  2.08 │
│         GPU Temperature (°C) │    38.87 │    33.00 │    41.00 │    41.00 │    41.00 │    39.00 │  2.38 │
│           XID Errors (count) │     0.00 │     0.00 │     0.00 │     0.00 │     0.00 │     0.00 │  0.00 │
└──────────────────────────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴───────┘
```

## Example CSV Export

```
Endpoint,GPU_Index,GPU_Name,GPU_UUID,Metric,avg,min,max,p1,p5,p10,p25,p50,p75,p90,p95,p99,std
localhost:9401,0,NVIDIA H100 80GB HBM3,GPU-afc3c15a-48a5-d669-0634-191c629f95fa,GPU Power Usage (W),348.69,120.57,386.02,120.57,120.57,,378.34,378.34,386.02,386.02,386.02,386.02,85.97
localhost:9401,0,NVIDIA H100 80GB HBM3,GPU-afc3c15a-48a5-d669-0634-191c629f95fa,Energy Consumption (MJ),0.24,0.23,0.25,0.23,0.23,,0.23,0.23,0.25,0.25,0.25,0.25,0.01
localhost:9401,0,NVIDIA H100 80GB HBM3,GPU-afc3c15a-48a5-d669-0634-191c629f95fa,GPU Utilization (%),45.82,0.00,66.00,0.00,0.00,,27.00,66.00,66.00,66.00,66.00,66.00,24.52
localhost:9401,0,NVIDIA H100 80GB HBM3,GPU-afc3c15a-48a5-d669-0634-191c629f95fa,Memory Copy Utilization (%),21.10,0.00,29.00,0.00,0.00,,15.00,29.00,29.00,29.00,29.00,29.00,10.11
localhost:9401,0,NVIDIA H100 80GB HBM3,GPU-afc3c15a-48a5-d669-0634-191c629f95fa,GPU Memory Used (GB),92.70,92.70,92.70,92.70,92.70,,92.70,92.70,92.70,92.70,92.70,92.70,0.00
localhost:9401,0,NVIDIA H100 80GB HBM3,GPU-afc3c15a-48a5-d669-0634-191c629f95fa,GPU Memory Free (GB),9.39,9.39,9.39,9.39,9.39,,9.39,9.39,9.39,9.39,9.39,9.39,0.00
localhost:9401,0,NVIDIA H100 80GB HBM3,GPU-afc3c15a-48a5-d669-0634-191c629f95fa,SM Clock Frequency (MHz),1980.00,1980.00,1980.00,1980.00,1980.00,,1980.00,1980.00,1980.00,1980.00,1980.00,1980.00,0.00
localhost:9401,0,NVIDIA H100 80GB HBM3,GPU-afc3c15a-48a5-d669-0634-191c629f95fa,Memory Clock Frequency (MHz),2619.00,2619.00,2619.00,2619.00,2619.00,,2619.00,2619.00,2619.00,2619.00,2619.00,2619.00,0.00
localhost:9401,0,NVIDIA H100 80GB HBM3,GPU-afc3c15a-48a5-d669-0634-191c629f95fa,Memory Temperature (°C),45.99,41.00,48.00,41.00,41.00,,46.00,46.00,48.00,48.00,48.00,48.00,2.08
localhost:9401,0,NVIDIA H100 80GB HBM3,GPU-afc3c15a-48a5-d669-0634-191c629f95fa,GPU Temperature (°C),38.87,33.00,41.00,33.00,33.00,,39.00,39.00,41.00,41.00,41.00,41.00,2.38
localhost:9401,0,NVIDIA H100 80GB HBM3,GPU-afc3c15a-48a5-d669-0634-191c629f95fa,XID Errors (count),0.00,0.00,0.00,0.00,0.00,,0.00,0.00,0.00,0.00,0.00,0.00,0.00
```

## Example JSON Export

```json
"telemetry_data": {
    "summary": {
      "endpoints_configured": [
        "http://localhost:9401/metrics"
      ],
      "endpoints_successful": [
        "http://localhost:9401/metrics"
      ],
      "start_time": "2025-10-13T01:48:03.689885",
      "end_time": "2025-10-13T01:48:55.971544"
    },
    "endpoints": {
      "localhost:9401": {
        "gpus": {
          "gpu_0": {
            "gpu_index": 0,
            "gpu_name": "NVIDIA H100 80GB HBM3",
            "gpu_uuid": "GPU-afc3c15a-48a5-d669-0634-191c629f95fa",
            "hostname": "69450c620e4d",
            "metrics": {
              "gpu_power_usage": {
                "avg": 348.6908823529412,
                "min": 120.57,
                "max": 386.022,
                "p1": 120.57,
                "p5": 120.57,
                "p10": null,
                "p25": 378.343,
                "p50": 378.343,
                "p75": 386.022,
                "p90": 386.022,
                "p95": 386.022,
                "p99": 386.022,
                "std": 85.96769288258695,
                "count": 153,
                "unit": "W"
              },
              "energy_consumption": {
                "avg": 0.23782271866013072,
                "min": 0.229901671,
                "max": 0.246497393,
                "p1": 0.229901671,
                "p5": 0.229901671,
                "p10": null,
                "p25": 0.23499845600000002,
                "p50": 0.23499845600000002,
                "p75": 0.246497393,
                "p90": 0.246497393,
                "p95": 0.246497393,
                "p99": 0.246497393,
                "std": 0.005916380392210164,
                "count": 153,
                "unit": "MJ"
              },
              "gpu_utilization": {
                "avg": 45.8235294117647,
                "min": 0.0,
                "max": 66.0,
                "p1": 0.0,
                "p5": 0.0,
                "p10": null,
                "p25": 27.0,
                "p50": 66.0,
                "p75": 66.0,
                "p90": 66.0,
                "p95": 66.0,
                "p99": 66.0,
                "std": 24.51706559093709,
                "count": 153,
                "unit": "%"
              },
              "memory_copy_utilization": {
                "avg": 21.098039215686274,
                "min": 0.0,
                "max": 29.0,
                "p1": 0.0,
                "p5": 0.0,
                "p10": null,
                "p25": 15.0,
                "p50": 29.0,
                "p75": 29.0,
                "p90": 29.0,
                "p95": 29.0,
                "p99": 29.0,
                "std": 10.109702002863262,
                "count": 153,
                "unit": "%"
              },
              "gpu_memory_used": {
                "avg": 92.69685977516342,
                "min": 92.69621555200001,
                "max": 92.698312704,
                "p1": 92.69621555200001,
                "p5": 92.69621555200001,
                "p10": null,
                "p25": 92.69621555200001,
                "p50": 92.69621555200001,
                "p75": 92.698312704,
                "p90": 92.698312704,
                "p95": 92.698312704,
                "p99": 92.698312704,
                "std": 0.0009674763104592773,
                "count": 153,
                "unit": "GB"
              },
              "gpu_memory_free": {
                "avg": 9.387256704836602,
                "min": 9.385803776000001,
                "max": 9.387900928,
                "p1": 9.385803776000001,
                "p5": 9.385803776000001,
                "p10": null,
                "p25": 9.385803776000001,
                "p50": 9.387900928,
                "p75": 9.387900928,
                "p90": 9.387900928,
                "p95": 9.387900928,
                "p99": 9.387900928,
                "std": 0.0009674763104633748,
                "count": 153,
                "unit": "GB"
              },
              "sm_clock_frequency": {
                "avg": 1980.0,
                "min": 1980.0,
                "max": 1980.0,
                "p1": 1980.0,
                "p5": 1980.0,
                "p10": null,
                "p25": 1980.0,
                "p50": 1980.0,
                "p75": 1980.0,
                "p90": 1980.0,
                "p95": 1980.0,
                "p99": 1980.0,
                "std": 0.0,
                "count": 153,
                "unit": "MHz"
              },
              "memory_clock_frequency": {
                "avg": 2619.0,
                "min": 2619.0,
                "max": 2619.0,
                "p1": 2619.0,
                "p5": 2619.0,
                "p10": null,
                "p25": 2619.0,
                "p50": 2619.0,
                "p75": 2619.0,
                "p90": 2619.0,
                "p95": 2619.0,
                "p99": 2619.0,
                "std": 0.0,
                "count": 153,
                "unit": "MHz"
              },
              "memory_temperature": {
                "avg": 45.99346405228758,
                "min": 41.0,
                "max": 48.0,
                "p1": 41.0,
                "p5": 41.0,
                "p10": null,
                "p25": 46.0,
                "p50": 46.0,
                "p75": 48.0,
                "p90": 48.0,
                "p95": 48.0,
                "p99": 48.0,
                "std": 2.081655738762016,
                "count": 153,
                "unit": "°C"
              },
              "gpu_temperature": {
                "avg": 38.869281045751634,
                "min": 33.0,
                "max": 41.0,
                "p1": 33.0,
                "p5": 33.0,
                "p10": null,
                "p25": 39.0,
                "p50": 39.0,
                "p75": 41.0,
                "p90": 41.0,
                "p95": 41.0,
                "p99": 41.0,
                "std": 2.383748929780352,
                "count": 153,
                "unit": "°C"
              },
              "xid_errors": {
                "avg": 0.0,
                "min": 0.0,
                "max": 0.0,
                "p1": 0.0,
                "p5": 0.0,
                "p10": null,
                "p25": 0.0,
                "p50": 0.0,
                "p75": 0.0,
                "p90": 0.0,
                "p95": 0.0,
                "p99": 0.0,
                "std": 0.0,
                "count": 153,
                "unit": "count"
              }
            }
          }
        }
      }
    }
  }
```
