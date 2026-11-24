---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: GPU Telemetry with AIPerf
---

# GPU Telemetry with AIPerf

This guide shows you how to collect GPU metrics (power, utilization, memory, temperature, etc.) during AIPerf benchmarking. GPU telemetry provides insights into GPU performance and resource usage while running inference workloads.

## Overview

This guide covers several setup paths depending on your inference backend and hardware:

### Path 1: Dynamo (Built-in DCGM)
If you're using **Dynamo**, it comes with DCGM pre-configured on port 9401. No additional setup needed! Just use the `--gpu-telemetry` flag to enable console display and optionally add additional DCGM url endpoints. URLs can be specified with or without the `http://` prefix (e.g., `localhost:9400` or `http://localhost:9400`).

### Path 2: Other Inference Servers (Custom DCGM)
If you're using **any other inference backend**, you'll need to set up DCGM separately.

### Path 3: Local GPU Monitoring (pynvml)
If you want **simple local GPU monitoring without DCGM**, use `--gpu-telemetry pynvml`. This uses NVIDIA's nvidia-ml-py Python library (commonly known as pynvml) to collect metrics directly from the GPU driver. No HTTP endpoints or additional containers required.

### Path 4: AMD ROCm GPUs (amdsmi)
If you're benchmarking against an inference server running on **AMD ROCm GPUs** (Instinct MI300X, MI355X, etc.), use `--gpu-telemetry amdsmi`. This uses the `amdsmi` Python bindings shipped with ROCm to collect metrics directly from the AMD driver. No HTTP endpoints required. Install the bindings via `pip install /opt/rocm/share/amd_smi/amdsmi-*.whl` if not already present (they ship with ROCm).

### Path 6: Huawei Ascend NPUs via npu-exporter (npu_exporter)
If the cluster already runs the Ascend **npu-exporter** DaemonSet (it ships with MindCluster / the Ascend device plugin), use `--gpu-telemetry npu_exporter <metrics-url>`. This scrapes `npu_chip_info_*` over HTTP, so AIPerf needs neither the `npu-smi` binary nor an NPU allocation of its own - the only Ascend source that works when every NPU is already allocated to inference pods.

### Path 5: Huawei Ascend NPUs (npu_smi)
If you're benchmarking against an inference server running on **Huawei Ascend NPUs** (910B/910C training cards, 310P/300I inference cards, etc.), use `--gpu-telemetry npu_smi`. This shells out to the `npu-smi` CLI that ships with the Ascend driver (CANN), so no exporter container, CANN toolkit, or Python binding is needed.

## Prerequisites

- NVIDIA GPU with CUDA support, **or** AMD GPU with ROCm 6.x/7.x, **or** Huawei Ascend NPU with the CANN driver (`npu-smi` on `PATH`)
- Docker installed and configured

## Understanding GPU Telemetry in AIPerf

AIPerf provides GPU telemetry collection with the `--gpu-telemetry` flag. Here's how it works:

DCGM collection runs on a fixed interval. If the previous scrape is still
fetching or processing, the collector skips that tick instead of queuing another
scrape. Slow endpoints can therefore produce fewer samples than the configured
interval would suggest.

### How the `--gpu-telemetry` Flag Works

| Usage | Command | What Gets Collected (If Available) | Console Display | Dashboard View | CSV/JSON Export |
|-------|---------|---------------------|-----------------|----------------|-----------------|
| **No flag** | `aiperf profile --model MODEL ...` | `http://localhost:9400/metrics` + `http://localhost:9401/metrics` | ❌ No | ❌ No | ✅ Yes |
| **Flag only** | `aiperf profile --model MODEL ... --gpu-telemetry` | `http://localhost:9400/metrics` + `http://localhost:9401/metrics` | ✅ Yes | ❌ No | ✅ Yes |
| **Dashboard mode** | `aiperf profile --model MODEL ... --gpu-telemetry dashboard` | `http://localhost:9400/metrics` + `http://localhost:9401/metrics` | ✅ Yes | ✅ Yes ([see dashboard](#real-time-dashboard-view)) | ✅ Yes |
| **Custom URLs** | `aiperf profile --model MODEL ... --gpu-telemetry node1:9400 http://node2:9400/metrics` | `http://localhost:9400/metrics` + `http://localhost:9401/metrics` + [custom URLs](#multi-node-gpu-telemetry-example) | ✅ Yes | ❌ No | ✅ Yes |
| **Dashboard + URLs** | `aiperf profile --model MODEL ... --gpu-telemetry dashboard localhost:9400` | `http://localhost:9400/metrics` + `http://localhost:9401/metrics` + [custom URLs](#multi-node-gpu-telemetry-example) | ✅ Yes | ✅ Yes ([see dashboard](#real-time-dashboard-view)) | ✅ Yes |
| **Custom metrics** | `aiperf profile --model MODEL ... --gpu-telemetry custom_gpu_metrics.csv` | `http://localhost:9400/metrics` + `http://localhost:9401/metrics` + [custom metrics from CSV](#customizing-displayed-metrics) | ✅ Yes | ❌ No | ✅ Yes |
| **pynvml mode** | `aiperf profile --model MODEL ... --gpu-telemetry pynvml` | Local GPUs via pynvml library ([see pynvml section](#3-using-pynvml-local-gpu-monitoring)) | ✅ Yes | ❌ No | ✅ Yes |
| **pynvml + dashboard** | `aiperf profile --model MODEL ... --gpu-telemetry pynvml dashboard` | Local GPUs via pynvml library | ✅ Yes | ✅ Yes ([see dashboard](#real-time-dashboard-view)) | ✅ Yes |
| **amdsmi mode** | `aiperf profile --model MODEL ... --gpu-telemetry amdsmi` | Local AMD ROCm GPUs via amdsmi library | ✅ Yes | ❌ No | ✅ Yes |
| **npu_smi mode** | `aiperf profile --model MODEL ... --gpu-telemetry npu_smi` | Local Ascend NPUs via the npu-smi CLI ([see npu-smi section](#5-using-npu_smi-huawei-ascend-npu-monitoring)) | ✅ Yes | ❌ No | ✅ Yes |
| **npu_exporter mode** | `aiperf profile --model MODEL ... --gpu-telemetry npu_exporter node1:8082/metrics` | Ascend NPUs via the npu-exporter Prometheus endpoint ([see npu-exporter section](#6-using-npu_exporter-ascend-npu-via-prometheus-exporter)) | ✅ Yes | ❌ No | ✅ Yes |
| **npu_smi + dashboard** | `aiperf profile --model MODEL ... --gpu-telemetry npu_smi dashboard` | Local Ascend NPUs via the npu-smi CLI | ✅ Yes | ✅ Yes ([see dashboard](#real-time-dashboard-view)) | ✅ Yes |
| **Disabled** | `aiperf profile --model MODEL ... --no-gpu-telemetry` | None | ❌ No | ❌ No | ❌ No |

> [!WARNING]
> **DCGM mode (default):** The default endpoints `http://localhost:9400/metrics` and `http://localhost:9401/metrics` are always attempted for telemetry collection, regardless of whether the `--gpu-telemetry` flag is used. The flag primarily controls whether metrics are displayed on the console and allows you to specify additional custom DCGM exporter endpoints.
>
> **pynvml mode:** When using `--gpu-telemetry pynvml`, DCGM endpoints are NOT used. Metrics are collected directly from local GPUs via the nvidia-ml-py library.
>
> **Platform naming:** NVIDIA metrics from DCGM and pynvml are emitted under `nvidia_*` field names, AMD metrics from amdsmi under `amd_*`, and Huawei Ascend metrics from npu_smi under `ascend_*`. Final GPU summaries include a `platform` field (`nvidia`, `amd`, or `ascend`). Metric semantics are platform-specific; do not compare telemetry across platforms without validating the workload, collector behavior, and metric definitions.
>
> **npu_exporter mode:** When using `--gpu-telemetry npu_exporter`, DCGM endpoints are NOT used. Metrics come from scraping each node's npu-exporter (`http://<node>:8082/metrics`) and are emitted under the same `ascend_*` field names as npu_smi. Each node runs its own exporter, so pass **one URL per node**; no `npu-smi` binary, driver mount, or NPU allocation is required inside AIPerf. The exporter refreshes every `-updateTime` seconds (5 s by default), so resolution is coarser than npu_smi's 1 s floor.
>
> **npu_smi mode:** When using `--gpu-telemetry npu_smi`, DCGM endpoints are NOT used. Metrics are collected from local Ascend NPUs by invoking `npu-smi` and are emitted under vendor-namespaced `ascend_*` field names (`ascend_power`, `ascend_temperature`, etc.). Because each scrape forks one `npu-smi` process per NPU, the collection interval is floored at 1 second; `npu-smi` exposes no cumulative energy counter, so Ascend energy metrics come from integrating the power gauge over the profiling window.
>
> **amdsmi mode:** When using `--gpu-telemetry amdsmi`, DCGM endpoints are NOT used. Metrics are collected directly from local AMD GPUs via the amdsmi library and emitted under vendor-namespaced `amd_*` field names (`amd_power`, `amd_gfx_activity`, `amd_temperature`, etc.) rather than NVML-shaped names. On Instinct datacenter parts `amd_mm_activity` is generally absent (sensor returns `'N/A'`); `amd_throttle_status` is a 0.0/1.0 snapshot per scrape (amdsmi exposes a boolean state, not a duration counter).
>
> To completely disable GPU telemetry collection, use `--no-gpu-telemetry`.

> [!NOTE]
> When specifying custom DCGM exporter URLs, the `http://` prefix is optional. URLs like `localhost:9400` will automatically be treated as `http://localhost:9400`. Both formats work identically.

> [!TIP]
> For simple local GPU monitoring without DCGM setup, use `--gpu-telemetry pynvml`. This collects metrics directly from the NVIDIA driver using the nvidia-ml-py library. See [Path 3: pynvml](#3-using-pynvml-local-gpu-monitoring) for details.

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

uv venv --python 3.11

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


GPU telemetry platform: nvidia. Metric semantics are platform-specific; cross-platform comparisons require workload and collector validation.

                          AIPerf | GPU Telemetry Summary
                               1/1 telemetry sources reachable
                                    • localhost:9401 ✔

                      localhost:9401 | GPU 0 | NVIDIA H100 80GB HBM3
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━┓
┃                       Metric ┃      avg ┃      min ┃      max ┃      p99 ┃      p90 ┃      p50 ┃   std ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━┩
│   NVIDIA GPU Power Usage (W) │   348.69 │   120.57 │   386.02 │   386.02 │   386.02 │   378.34 │ 85.97 │
│ NVIDIA Energy Consumption (MJ) │     0.24 │     0.23 │     0.25 │     0.25 │     0.25 │     0.23 │  0.01 │
│   NVIDIA GPU Utilization (%) │    45.82 │     0.00 │    66.00 │    66.00 │    66.00 │    66.00 │ 24.52 │
│  NVIDIA Memory Utilization (%) │    21.10 │     0.00 │    29.00 │    29.00 │    29.00 │    29.00 │ 10.11 │
│  NVIDIA GPU Memory Used (GB) │    92.70 │    92.70 │    92.70 │    92.70 │    92.70 │    92.70 │  0.00 │
│   Framebuffer Memory Free (MB) │ 9,387.26 │ 9,385.80 │ 9,387.90 │ 9,387.90 │ 9,387.90 │ 9,387.90 │  0.97 │
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

uv venv --python 3.11

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

---

# 3: Using pynvml (Local GPU Monitoring)

For simple local GPU monitoring without DCGM infrastructure, AIPerf supports direct GPU metrics collection using NVIDIA's nvidia-ml-py Python library (commonly known as pynvml). This approach requires no additional containers, HTTP endpoints, or DCGM setup.

## Prerequisites

- NVIDIA GPU with driver installed
- nvidia-ml-py package: `pip install nvidia-ml-py`

## When to Use pynvml

| Scenario | Recommended Approach |
|----------|---------------------|
| Local development/testing | pynvml |
| Single-node inference server | pynvml or DCGM |
| Multi-node distributed setup | DCGM (HTTP endpoints required) |
| Production with existing DCGM | DCGM |
| Quick GPU monitoring without setup | pynvml |

## Run AIPerf with pynvml

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
    --gpu-telemetry pynvml
```

> [!TIP]
> Add `dashboard` after `pynvml` for the real-time terminal UI: `--gpu-telemetry pynvml dashboard`

## Metrics Collected via pynvml

The nvidia-ml-py library (pynvml) collects the following metrics directly from the NVIDIA driver:

| Metric | Description | Unit |
|--------|-------------|------|
| `nvidia_power_usage` | Current power draw | W |
| `nvidia_energy_consumption` | Total energy since boot (see note below) | MJ |
| `nvidia_gpu_utilization` | GPU compute utilization | % |
| `nvidia_memory_utilization` | Memory controller utilization | % |
| `nvidia_memory_used` | Framebuffer memory in use | GB |
| `nvidia_temperature` | GPU die temperature | °C |
| `nvidia_sm_utilization` | Streaming multiprocessor utilization | % |
| `nvidia_decoder_utilization` | Video decoder utilization | % |
| `nvidia_encoder_utilization` | Video encoder utilization | % |
| `nvidia_jpg_utilization` | JPEG decoder utilization | % |
| `nvidia_power_violation` | Throttling duration due to power limits | µs |

> [!NOTE]
> Not all metrics are available on all GPU models. AIPerf gracefully handles missing metrics and reports only what the hardware supports.

> [!NOTE]
> `nvidia_energy_consumption` needs the NVML total-energy counter, which requires Volta or
> newer and a driver that exports `nvmlDeviceGetTotalEnergyConsumption`. AIPerf probes it
> once at startup; where it is absent the field is omitted for that GPU and a warning names
> the requirement, rather than the run finishing silently without energy. Power efficiency
> metrics are still produced, derived from the power gauge instead. See
> [Energy source](../metrics-reference.md#gpu-power-efficiency-metrics).

## Comparing DCGM vs pynvml

| Feature | DCGM | pynvml |
|---------|------|--------|
| Setup complexity | Requires container/service | Just install nvidia-ml-py Python package |
| Multi-node support | Yes (via HTTP endpoints) | No (local only) |
| Metrics granularity | High (profiling-level metrics) | Standard (driver-level metrics) |
| Kubernetes integration | Native with dcgm-exporter | Not applicable |
| XID error reporting | Yes | No |

---

## 4. Using amdsmi (Local AMD ROCm GPU Monitoring)

For inference workloads on **AMD Instinct GPUs** (MI300X, MI355X, etc.), use `--gpu-telemetry amdsmi`. This collects metrics directly from local AMD GPUs via the `amdsmi` Python library shipped with ROCm.

### When to Use amdsmi

- Benchmarking against vLLM-ROCm, SGLang-ROCm, TGI, or any ROCm-backed inference server running on the same machine as AIPerf.
- Local single-node monitoring with no need for HTTP exporters.

### Run AIPerf with amdsmi

```bash
aiperf profile \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --endpoint-type chat \
    --url http://localhost:8000 \
    --concurrency 16 --request-count 200 \
    --gpu-telemetry amdsmi
```

### Metrics Collected via amdsmi

AMD signals are emitted under their own vendor-namespaced field names (not aliased onto NVML-shaped names) because the underlying sensors do not always measure the same physical quantity (e.g. amdsmi `gfx_activity` and NVML `sm_utilization` sample at different scopes).

| Metric | Source | Notes |
|---|---|---|
| `amd_power` (W) | `amdsmi_get_power_info().current_socket_power` | Already in W; no scaling. Falls back to `average_socket_power` if `current_socket_power` is `'N/A'`. |
| `amd_energy_consumption` (MJ) | `amdsmi_get_energy_count()` | `accumulator * counter_resolution` (µJ) → MJ. Cumulative counter — accumulator computes a delta against the pre-profile baseline. Reads `energy_accumulator` first; falls back to the older `power` field name on ROCm < 6.2. |
| `amd_gfx_activity` (%) | `amdsmi_get_gpu_activity().gfx_activity` | Graphics engine activity. |
| `amd_umc_activity` (%) | `amdsmi_get_gpu_activity().umc_activity` | Memory controller activity. |
| `amd_mm_activity` (%) | `amdsmi_get_gpu_activity().mm_activity` | Multimedia engine activity. Generally `'N/A'` on Instinct datacenter GPUs — field will be absent rather than emitted as zero. |
| `amd_memory_used` (GB) | `amdsmi_get_gpu_memory_usage(VRAM)` | bytes → GB. |
| `amd_temperature` (°C) | `amdsmi_get_temp_metric(JUNCTION)` | Falls back to HOTSPOT. EDGE is unsupported on Instinct GPUs. Unit conversion is gated on `amdsmi.__version__` (≥ 26.x already returns Celsius; older bindings return millidegrees and are divided by 1000). |
| `amd_ecc_uncorrectable` | `amdsmi_get_gpu_total_ecc_count().uncorrectable_count` | Cumulative uncorrectable ECC error count. Counter — accumulator computes a delta. |
| `amd_throttle_status` | `amdsmi_get_gpu_metrics_info().throttle_status` (and `indep_throttle_status`) | 0.0/1.0 snapshot per scrape — 1.0 if any throttle indicator is active. amdsmi exposes a state (bool/bitfield), not a duration counter; a fraction-throttled summary can be derived from the average. Field is left absent when both signals return `'N/A'` (sensor unsupported), so "unsupported" is not silently reported as "not throttled". |

> **Plotting AMD activity:** the dual-axis "throughput + GPU utilization" plot prefers `nvidia_gpu_utilization` and falls back to `amd_gfx_activity` on AMD-only runs, labeling the axis by the actual signal ("AMD GFX Activity"). Because NVIDIA SM-occupancy utilization and AMD GFX activity are distinct physical signals, the plot always carries a subtitle noting the values are not comparable across vendors.

See [Comparing DCGM vs pynvml vs amdsmi vs npu_smi](#comparing-dcgm-vs-pynvml-vs-amdsmi-vs-npu_smi) below for the full matrix.

---

## 5. Using npu_smi (Huawei Ascend NPU Monitoring)

For inference workloads on **Huawei Ascend NPUs**, use `--gpu-telemetry npu_smi`. This collects metrics from local Ascend devices by invoking `npu-smi`, the device management CLI that ships with the Ascend driver (CANN). No exporter, CANN toolkit, or Python binding is required.

### When to Use npu_smi

- Benchmarking against vLLM-Ascend, MindIE, or any inference server running on the same machine as AIPerf.
- Ascend hosts where no DCGM-style Prometheus exporter is deployed.
- Local single-node monitoring.

### Run AIPerf with npu_smi

```bash
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --url http://localhost:8000 \
    --concurrency 16 --request-count 200 \
    --gpu-telemetry npu_smi
```

Add `dashboard` for the real-time terminal UI: `--gpu-telemetry npu_smi dashboard`.

In a container, `npu-smi` and `/dev/davinci*` must be visible to the AIPerf process (the same device exposure the inference server uses). The collector fails fast at config time with `npu-smi not found on PATH` if the CLI is missing, rather than silently collecting nothing.

### Metrics Collected via npu_smi

Signals are emitted under vendor-namespaced `ascend_*` field names. They come from the machine-readable `npu-smi info -t <type> -i <id>` subcommands; the human-facing `npu-smi info` summary table is only used as a last-resort device-discovery fallback because its column layout varies by product line and driver release.

| Metric | Source | Notes |
|---|---|---|
| `ascend_power` (W) | `npu-smi info -t common` → `NPU Real-time Power` | Falls back to `Power`. Rated/limit keys (e.g. `Rated Power Dissipation`) are never used as the live draw. |
| `ascend_temperature` (°C) | `-t common` → `Temperature` (falls back to `NPU Temperature`, `HBM Temperature`) | Chip temperature as reported by the driver. |
| `ascend_aicore_utilization` (%) | `-t common` → `Aicore Usage Rate` | AI Core occupancy. Not the same physical signal as NVML `sm_utilization` or AMD `gfx_activity`. |
| `ascend_memory_utilization` (%) | `-t common` → `HBM Usage Rate` | Falls back to `Memory Usage Rate`. |
| `ascend_memory_used` (GB) | `-t memory` → `HBM Capacity` × usage rate | Capacity is read once at init; MiB → GB. Absent when the driver reports no capacity. |
| `ascend_health` | `-t common` → `Chip Health` | `1.0` when the driver reports `OK`, `0.0` for any warning/alarm state. |

Signals the driver does not report stay absent instead of defaulting to zero, so an unsupported sensor is never mistaken for a real reading.

### npu_smi Behavior Notes

- **One record per NPU card.** `npu-smi info -m` lists both the AI chip and the MCU management chip per card; MCU rows are skipped. `gpu_uuid` is `NPU-<card id>` because `npu-smi` exposes no device UUID (unlike NVML's `GPU-<uuid>`), and `gpu_index` is the card ID used with `-i`.
- **Collection interval is floored at 1 second.** Each scrape forks one `npu-smi` process per NPU, so the collector raises `AIPERF_GPU_COLLECTION_INTERVAL` values below 1s and logs the change once at init. Raising the interval further is still honored.
- **Layered fallback.** `npu-smi info -t common` batches power/temperature/usage in one call and is probed once at init. On drivers that do not implement it, the collector falls back to `-t usages`, `-t power`, and `-t temp` per scrape. A transient `-t common` failure falls back for that tick only.
- **No energy counter.** `npu-smi` reports instantaneous power only, so there is no `ascend_energy_consumption` field. When power efficiency metrics are enabled, Ascend energy is derived by integrating `ascend_power` over the profiling window (the same power-integration path used for any vendor without an energy counter).
- **Multi-die parts.** On cards that present more than one AI chip (e.g. 910C), the collector emits the card-level values that `npu-smi info -t common -i <id>` reports for chip 0, which the driver documents as the NPU total power.

### Comparing DCGM vs pynvml vs amdsmi vs npu_smi

| Feature | DCGM | pynvml | amdsmi | npu_smi |
|---|---|---|---|---|
| Hardware | NVIDIA | NVIDIA | AMD ROCm | Huawei Ascend |
| Setup complexity | Requires container/service | `pip install nvidia-ml-py` | Ships with ROCm; install wheel from `/opt/rocm/share/amd_smi/` if missing | Ships with the Ascend driver (`npu-smi` on `PATH`) |
| Multi-node support | Yes (HTTP) | No (local) | No (local) | No (local) |
| Field naming | `nvidia_*` | `nvidia_*` | `amd_*` | `ascend_*` |
| Encoder/decoder util | Yes | Yes | No (Instinct GPUs report `'N/A'`) | No |
| Error reporting | XID errors | (none) | ECC uncorrectable count | Health flag (`ascend_health`) |
| SM-level utilization | Yes (DCGM_FI_PROF_SM_ACTIVE) | Yes (GPM API) | Aliased to `gfx_activity` | AI Core occupancy (`ascend_aicore_utilization`) |
| Cumulative energy counter | Yes | Yes (Volta+) | Yes | No (power integration) |

---

## 6. Using npu_exporter (Ascend NPU via Prometheus Exporter)

When the cluster already deploys the Ascend **npu-exporter** (a DaemonSet that ships with MindCluster / the Ascend device plugin), AIPerf can scrape it over HTTP instead of invoking `npu-smi`. This is the only Ascend source that works when **every NPU is already allocated to inference pods** - npu_smi would need an NPU allocation itself and would otherwise sit `Pending`.

### When to Use npu_exporter

- The cluster runs `npu-exporter` (check for a DaemonSet named `npu-exporter` in the monitoring namespace).
- AIPerf runs as a separate pod/container that cannot or should not take NPU devices.
- Multi-node coverage is wanted from one AIPerf run.

### Run AIPerf with npu_exporter

```bash
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --url http://localhost:8000 \
    --concurrency 16 --request-count 200 \
    --gpu-telemetry npu_exporter 10.233.224.133:8082/metrics 10.233.218.121:8082/metrics
```

Each node's exporter publishes only that node's chips, so pass one URL per node. The `http://` prefix is optional. In a YAML config or an `AIPerfJob` CR, the same reads as:

```yaml
gpuTelemetry:
  enabled: true
  collector: npu_exporter
  urls:
    - http://10.233.224.133:8082/metrics
    - http://10.233.218.121:8082/metrics
```

> A bare Service IP does not cover all nodes: a ClusterIP service load-balances each scrape to a single exporter pod. Use per-node Pod IPs (they change when the DaemonSet restarts), or ask the platform to add `hostPort: 8082` / `hostNetwork` to the npu-exporter DaemonSet so `http://<node-ip>:8082/metrics` is stable.

### Metrics Collected via npu_exporter

Same six `ascend_*` fields as npu_smi, mapped from these families:

| Field | npu-exporter family | Notes |
|---|---|---|
| `ascend_power` | `npu_chip_info_power` | Watts, as published. |
| `ascend_temperature` | `npu_chip_info_temperature` | °C. `npu_chip_info_hbm_temperature` is a separate HBM sensor and is not used for this field. |
| `ascend_aicore_utilization` | `npu_chip_info_utilization` | AI Core % (`overall_utilization` / `vector_utilization` are published too but not mapped). |
| `ascend_memory_utilization` | `npu_chip_info_hbm_utilization` | %. |
| `ascend_memory_used` | `npu_chip_info_hbm_used_memory` | MB -> GB. |
| `ascend_health` | `npu_chip_info_health_status` | `1` healthy, otherwise `0`. |

Identity comes from the exporter's labels: `vdie_id` is the GPU UUID, `id` is the chip index, plus `model_name`, `pcie_bus_info`, `pod_name`, `namespace`. Info-style families (`npu_chip_info_name`, `npu_chip_info_serial_number`) carry non-numeric values and are skipped without failing the scrape.

### npu_smi vs npu_exporter

| | npu_smi | npu_exporter |
|---|---|---|
| Needs `npu-smi` in the AIPerf container | Yes | No |
| Needs an NPU allocation | Yes | No |
| Resolution | 1 s floor | exporter update interval (5 s default) |
| Device identity | `NPU-<id>` (no UUID from npu-smi) | `vdie_id` UUID + pod/namespace |
| Multi-node | local host only | one URL per node |
| Works when all NPUs are allocated | No (Pending) | Yes |

---

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

**Behavior**: Custom metrics **extend** (not replace) the default NVIDIA metrics:
- `nvidia_power_usage`
- `nvidia_energy_consumption`
- `nvidia_gpu_utilization`
- `nvidia_memory_utilization`
- `nvidia_memory_used`
- `nvidia_temperature`
- `nvidia_encoder_utilization`
- `nvidia_decoder_utilization`
- `nvidia_sm_utilization`
- `nvidia_xid_errors`
- `nvidia_power_violation`

> [!NOTE]
> The file path can be absolute or relative. Use `.csv` extension so AIPerf can distinguish it from DCGM endpoint URLs.

> [!TIP]
> You can start with the example CSV from the vLLM setup section and customize it by commenting out metrics you don't need or adding new DCGM metrics.

## Example Console Display:

```
GPU telemetry platform: nvidia. Metric semantics are platform-specific; cross-platform comparisons require workload and collector validation.

                                  AIPerf | GPU Telemetry Summary
                                       1/1 telemetry sources reachable
                                            • localhost:9401 ✔

                              localhost:9401 | GPU 0 | NVIDIA H100 80GB HBM3
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━┓
┃                       Metric ┃      avg ┃      min ┃      max ┃      p99 ┃      p90 ┃      p50 ┃   std ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━┩
│   NVIDIA GPU Power Usage (W) │   348.69 │   120.57 │   386.02 │   386.02 │   386.02 │   378.34 │ 85.97 │
│ NVIDIA Energy Consumption (MJ) │     0.24 │     0.23 │     0.25 │     0.25 │     0.25 │     0.23 │  0.01 │
│   NVIDIA GPU Utilization (%) │    45.82 │     0.00 │    66.00 │    66.00 │    66.00 │    66.00 │ 24.52 │
│  NVIDIA Memory Utilization (%) │    21.10 │     0.00 │    29.00 │    29.00 │    29.00 │    29.00 │ 10.11 │
│  NVIDIA GPU Memory Used (GB) │    92.70 │    92.70 │    92.70 │    92.70 │    92.70 │    92.70 │  0.00 │
│   Framebuffer Memory Free (MB) │ 9,387.26 │ 9,385.80 │ 9,387.90 │ 9,387.90 │ 9,387.90 │ 9,387.90 │  0.97 │
│     SM Clock Frequency (MHz) │ 1,980.00 │ 1,980.00 │ 1,980.00 │ 1,980.00 │ 1,980.00 │ 1,980.00 │  0.00 │
│ Memory Clock Frequency (MHz) │ 2,619.00 │ 2,619.00 │ 2,619.00 │ 2,619.00 │ 2,619.00 │ 2,619.00 │  0.00 │
│      Memory Temperature (°C) │    45.99 │    41.00 │    48.00 │    48.00 │    48.00 │    46.00 │  2.08 │
│         GPU Temperature (°C) │    38.87 │    33.00 │    41.00 │    41.00 │    41.00 │    39.00 │  2.38 │
│           XID Errors (count) │     0.00 │     0.00 │     0.00 │     0.00 │     0.00 │     0.00 │  0.00 │
└──────────────────────────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴───────┘
```

## Example CSV Export

```
Endpoint,GPU_Index,GPU_Name,GPU_UUID,Platform,Metric,avg,min,max,p1,p5,p10,p25,p50,p75,p90,p95,p99,std
localhost:9401,0,NVIDIA H100 80GB HBM3,GPU-afc3c15a-48a5-d669-0634-191c629f95fa,nvidia,NVIDIA GPU Power Usage (W),348.69,120.57,386.02,120.57,120.57,,378.34,378.34,386.02,386.02,386.02,386.02,85.97
localhost:9401,0,NVIDIA H100 80GB HBM3,GPU-afc3c15a-48a5-d669-0634-191c629f95fa,nvidia,NVIDIA Energy Consumption (MJ),0.24,0.23,0.25,0.23,0.23,,0.23,0.23,0.25,0.25,0.25,0.25,0.01
localhost:9401,0,NVIDIA H100 80GB HBM3,GPU-afc3c15a-48a5-d669-0634-191c629f95fa,nvidia,NVIDIA GPU Utilization (%),45.82,0.00,66.00,0.00,0.00,,27.00,66.00,66.00,66.00,66.00,66.00,24.52
localhost:9401,0,NVIDIA H100 80GB HBM3,GPU-afc3c15a-48a5-d669-0634-191c629f95fa,nvidia,NVIDIA Memory Utilization (%),21.10,0.00,29.00,0.00,0.00,,15.00,29.00,29.00,29.00,29.00,29.00,10.11
localhost:9401,0,NVIDIA H100 80GB HBM3,GPU-afc3c15a-48a5-d669-0634-191c629f95fa,nvidia,NVIDIA GPU Memory Used (GB),92.70,92.70,92.70,92.70,92.70,,92.70,92.70,92.70,92.70,92.70,92.70,0.00
localhost:9401,0,NVIDIA H100 80GB HBM3,GPU-afc3c15a-48a5-d669-0634-191c629f95fa,nvidia,Framebuffer Memory Free (MB),9387.26,9385.80,9387.90,9385.80,9385.80,,9385.80,9387.90,9387.90,9387.90,9387.90,9387.90,0.97
localhost:9401,0,NVIDIA H100 80GB HBM3,GPU-afc3c15a-48a5-d669-0634-191c629f95fa,nvidia,SM Clock Frequency (MHz),1980.00,1980.00,1980.00,1980.00,1980.00,,1980.00,1980.00,1980.00,1980.00,1980.00,1980.00,0.00
localhost:9401,0,NVIDIA H100 80GB HBM3,GPU-afc3c15a-48a5-d669-0634-191c629f95fa,nvidia,Memory Clock Frequency (MHz),2619.00,2619.00,2619.00,2619.00,2619.00,,2619.00,2619.00,2619.00,2619.00,2619.00,2619.00,0.00
localhost:9401,0,NVIDIA H100 80GB HBM3,GPU-afc3c15a-48a5-d669-0634-191c629f95fa,nvidia,Memory Temperature (°C),45.99,41.00,48.00,41.00,41.00,,46.00,46.00,48.00,48.00,48.00,48.00,2.08
localhost:9401,0,NVIDIA H100 80GB HBM3,GPU-afc3c15a-48a5-d669-0634-191c629f95fa,nvidia,NVIDIA GPU Temperature (°C),38.87,33.00,41.00,33.00,33.00,,39.00,39.00,41.00,41.00,41.00,41.00,2.38
localhost:9401,0,NVIDIA H100 80GB HBM3,GPU-afc3c15a-48a5-d669-0634-191c629f95fa,nvidia,NVIDIA XID Errors (count),0.00,0.00,0.00,0.00,0.00,,0.00,0.00,0.00,0.00,0.00,0.00,0.00
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
            "platform": "nvidia",
            "hostname": "69450c620e4d",
            "metrics": {
              "nvidia_power_usage": {
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
              "nvidia_energy_consumption": {
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
              "nvidia_gpu_utilization": {
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
              "nvidia_memory_utilization": {
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
              "nvidia_memory_used": {
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
              "nvidia_fb_free": {
                "avg": 9387.256704836602,
                "min": 9385.803776000001,
                "max": 9387.900928,
                "p1": 9385.803776000001,
                "p5": 9385.803776000001,
                "p10": null,
                "p25": 9385.803776000001,
                "p50": 9387.900928,
                "p75": 9387.900928,
                "p90": 9387.900928,
                "p95": 9387.900928,
                "p99": 9387.900928,
                "std": 0.9674763104633748,
                "count": 153,
                "unit": "MB"
              },
              "nvidia_sm_clock": {
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
              "nvidia_mem_clock": {
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
              "nvidia_memory_temp": {
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
              "nvidia_temperature": {
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
              "nvidia_xid_errors": {
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
