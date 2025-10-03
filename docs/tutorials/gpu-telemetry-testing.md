<!--
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
<!--
Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
 * Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
 * Neither the name of NVIDIA CORPORATION nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
-->

# GPU Telemetry with AIPerf

This guide shows you how to collect GPU metrics (power, utilization, memory, temperature, etc.) during AIPerf benchmarking. GPU telemetry provides deep insights into GPU performance and resource usage while running inference workloads.

**What you'll learn:**
- How to set up NVIDIA DCGM Exporter for GPU metrics collection
- How to use the `--gpu-telemetry` flag in AIPerf
- How to collect metrics from single or multiple nodes
- Complete working examples for different inference backends

## Overview

This guide covers two setup paths depending on your inference backend:

### 🔷 Path 1: Dynamo (Built-in DCGM)
If you're using **Dynamo**, it comes with DCGM pre-configured on port 9401. No additional setup needed!

- ✅ No DCGM setup required
- ✅ Already on port 9401 (AIPerf's default)
- ✅ Just use `--gpu-telemetry` flag to enable console display

### 🔶 Path 2: Custom DCGM (vLLM, SGLang, TRT-LLM, etc.)
If you're using **any other inference backend**, you'll need to set up DCGM separately.

- 📦 Works with: vLLM, SGLang, TRT-LLM, OpenAI-compatible servers, etc.
- ⚙️ Requires: DCGM Exporter container setup
- 🔧 Configure: Forward to port 9401 (AIPerf's default)
- 🎯 Result: Same `--gpu-telemetry` flag usage as Dynamo

## Prerequisites

- NVIDIA GPU with CUDA support
- Docker installed and configured
- Access to NVIDIA container registry (`nvcr.io`)
- AIPerf installed or access to AIPerf container

## 📊 Understanding GPU Telemetry in AIPerf

AIPerf provides flexible GPU telemetry collection with the `--gpu-telemetry` flag. Here's how it works:

### How the `--gpu-telemetry` Flag Works

| Usage | Command | What Gets Collected (If Available) | Console Display | CSV/JSON Export |
|-------|---------|---------------------|-----------------|-----------------|
| **No flag** | `aiperf profile --model MODEL ...` | `http://localhost:9401/metrics` | ❌ No | ✅ Yes |
| **Flag only** | `aiperf profile --model MODEL ... --gpu-telemetry` | `http://localhost:9401/metrics` | ✅ Yes | ✅ Yes |
| **Custom URLs** | `aiperf profile --model MODEL ... --gpu-telemetry http://node1:9400/metrics http://node2:9400/metrics` | `http://localhost:9401/metrics` + custom URLs | ✅ Yes | ✅ Yes |

> [!IMPORTANT]
> The default endpoint `http://localhost:9401/metrics` is ALWAYS attempted for telemetry collection, regardless of whether the `--gpu-telemetry` flag is used. The flag primarily controls whether metrics are displayed on the console and allows you to specify additional custom DCGM exporter endpoints.

---

# 🔷 Path 1: Using Dynamo

Dynamo includes DCGM out of the box on port 9401 - no extra setup needed!

## Setup Dynamo Server

<!-- setup-dynamo-gpu-telemetry-openai-endpoint-server -->
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
timeout 900 bash -c 'while [ "$(curl -s -o /dev/null -w "%{http_code}" localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"Qwen/Qwen3-0.6B\",\"messages\":[{\"role\":\"user\",\"content\":\"a\"}],\"max_completion_tokens\":1}")" != "200" ]; do sleep 2; done' || { echo "Dynamo not ready after 15min"; exit 1; }

# Verify GPU metrics are being collected by Dynamo's DCGM
if ! curl -s localhost:9401/metrics | grep -q "DCGM_FI_DEV_GPU_UTIL"; then
    echo "GPU utilization metrics not found in Dynamo DCGM"
    exit 1
fi
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
    --gpu-telemetry default
```
<!-- /aiperf-run-dynamo-gpu-telemetry-openai-endpoint-server -->

---

# 🔶 Path 2: Using Any Other Backend

This path works with **vLLM, SGLang, TRT-LLM, or any inference server**. We'll use vLLM as an example.

## Step 1: Create Custom DCGM Metrics File

First, create a configuration file that tells DCGM which GPU metrics to collect.

<!-- setup-vllm-gpu-telemetry-openai-endpoint-server -->
```bash
# Create custom GPU metrics configuration file
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

# Memory usage
DCGM_FI_DEV_FB_USED, gauge, Framebuffer memory used (in MiB)

# Utilization
DCGM_FI_DEV_GPU_UTIL, gauge, GPU utilization (in %)
EOF
```

> [!TIP]
> You can customize this file by commenting out metrics you don't need. Lines starting with `#` are ignored.

## Step 2: Start DCGM Exporter

Now start the DCGM Exporter container with the custom metrics file:

```bash
# Set environment variables
export DCGM_EXPORTER_IMAGE="nvcr.io/nvidia/k8s/dcgm-exporter:4.2.0-4.1.0-ubuntu22.04"

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
```

**Key Configuration:**
- `-p 9401:9400` - Forward container's port 9400 to host's port 9401 (AIPerf's default)
- `-e DCGM_EXPORTER_INTERVAL=33` - Collect metrics every 33ms for fine-grained profiling
- `-v custom_gpu_metrics.csv:...` - Mount your custom metrics configuration

## Step 3: Start Your Inference Server

Start your inference backend (vLLM in this example):

```bash
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
# Check DCGM Exporter is running and GPU metrics are being collected
if ! curl -s localhost:9401/metrics | grep -q "DCGM_FI_DEV_GPU_UTIL"; then
    echo "GPU utilization metrics not found"
    exit 1
fi

# Check vLLM inference server is ready
timeout 900 bash -c 'while [ "$(curl -s -o /dev/null -w "%{http_code}" localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"Qwen/Qwen3-0.6B\",\"messages\":[{\"role\":\"user\",\"content\":\"test\"}],\"max_tokens\":1}")" != "200" ]; do sleep 2; done' || { echo "vLLM not ready after 15min"; exit 1; }
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
    --gpu-telemetry default
```
<!-- /aiperf-run-vllm-gpu-telemetry-openai-endpoint-server -->

---

## 🌐 Multi-Node GPU Telemetry Example

For distributed setups with multiple nodes, you can collect GPU telemetry from all nodes simultaneously:

```bash
# Example: Collecting telemetry from 3 nodes in a distributed setup
# Note: The default http://localhost:9401/metrics is always attempted in addition to these URLs
aiperf profile \
    --model "Qwen/Qwen3-0.6B" \
    --endpoint-type chat \
    --endpoint /v1/chat/completions \
    --url localhost:8000 \
    --gpu-telemetry http://node1:9400/metrics http://node2:9400/metrics http://node3:9400/metrics \
    --streaming \
    --concurrency 16 \
    --request-count 128 \
    --verbose
```

This will collect GPU metrics from:
- `http://localhost:9401/metrics` (default, always attempted)
- `http://node1:9400/metrics` (custom node 1)
- `http://node2:9400/metrics` (custom node 2)
- `http://node3:9400/metrics` (custom node 3)

All metrics are saved to the output CSV and JSON files, with GPU indices and hostnames distinguishing metrics from different nodes.

---

# 🧹 Cleanup

## Dynamo Cleanup
```bash
# Stop Dynamo server if running
if [ -f dynamo_server.pid ]; then
    kill $(cat dynamo_server.pid) 2>/dev/null || true
    rm -f dynamo_server.pid
fi

# Stop Docker Compose services
docker compose -f docker-compose.yml down 2>/dev/null || true

# Clean up generated files
rm -f dynamo_server.log
rm -f docker-compose.yml
rm -f aiperf_*.csv aiperf_*.json
rm -rf AIPerf/
```

## Custom DCGM Cleanup (vLLM, etc.)
```bash
# Stop and remove containers
docker stop dcgm-exporter vllm-server || true
docker rm dcgm-exporter vllm-server || true

# Clean up generated files
rm -f custom_gpu_metrics.csv
rm -f vllm_server.name
rm -rf vllm_gpu_telemetry_results
rm -f aiperf_*.csv aiperf_*.json
rm -rf AIPerf/
```

---

# 🧪 Advanced: CI Testing

For contributors and CI maintainers, this section explains how to run automated end-to-end tests.

### Prerequisites for Local Testing

1. **System Requirements:**
   - Linux system with NVIDIA GPU
   - Docker installed and configured
   - Python 3.8+ installed
   - Access to pull NVIDIA containers from `nvcr.io`

2. **Environment Setup:**
   ```bash
   # Clone the AIPerf repository
   git clone https://github.com/NVIDIA/AIPerf.git
   cd AIPerf

   # Install Python dependencies for the test framework
   pip install -r tests/ci/test_docs_end_to_end/requirements.txt  # if exists
   # OR install individual dependencies:
   pip install pathlib dataclasses
   ```

### Running the Tests

1. **Dry Run (Discover Commands Only):**
   ```bash
   cd tests/ci/test_docs_end_to_end
   python3 main.py --dry-run
   ```
   This will scan all markdown files and show discovered server configurations without executing them.

2. **Run All Server Tests:**
   ```bash
   cd tests/ci/test_docs_end_to_end
   python3 main.py --all-servers
   ```
   This will execute all discovered server setups including both GPU telemetry paths.

3. **Discovered Server Configurations:**
   The framework will automatically discover these servers from this document:
   - `dynamo-gpu-telemetry` - Dynamo path with existing DCGM at port 9401
   - `vllm-gpu-telemetry` - vLLM path with custom DCGM at port 9400

   Each server runs: setup → health check → AIPerf runs

4. **Monitor Test Execution:**
   ```bash
   # View real-time logs
   tail -f test_execution.log

   # Check for any running containers
   docker ps

   # View container logs if needed (for Dynamo path)
   docker logs $(docker ps -q --filter "ancestor=nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.4.0")

   # View container logs if needed (for vLLM path)
   docker logs dcgm-exporter
   docker logs vllm-server
   ```

### Understanding the Two Test Paths

#### Path 1: Dynamo GPU Telemetry Testing
- **Tags**: `dynamo-gpu-telemetry-endpoint-server`
- **DCGM Port**: 9401 (provided by Dynamo)
- **Inference Port**: 8080
- **Setup**: Uses Dynamo's built-in DCGM exporter
- **Use Case**: Testing with Dynamo-based inference servers

#### Path 2: Custom DCGM GPU Telemetry Testing (vLLM Example)
- **Tags**: `vllm-gpu-telemetry-endpoint-server`
- **DCGM Port**: 9401 (container port 9400 forwarded to host port 9401)
- **Inference Port**: 8000 (backend-dependent)
- **Setup**: Custom DCGM exporter + inference backend (vLLM in this example)
- **Use Case**: Testing with any inference backend (vLLM, SGLang, TRT-LLM, etc.)

### Test Framework Details

The end-to-end testing framework:

- **Parses** markdown files for tagged code blocks
- **Extracts** commands with patterns: `setup-*-endpoint-server`, `health-check-*-endpoint-server`, `aiperf-run-*-endpoint-server`
- **Executes** setup commands in background with monitoring
- **Validates** services using health check commands
- **Runs** AIPerf benchmarks with collected configurations
- **Cleans up** resources after test completion

### Troubleshooting Local Tests

1. **GPU Access Issues:**
   ```bash
   # Verify GPU is accessible
   nvidia-smi

   # Test Docker GPU access
   docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu20.04 nvidia-smi
   ```

2. **Port Conflicts:**
   ```bash
   # Check what's using ports 8000, 8080, 9400, and 9401
   netstat -tuln | grep -E ':8000|:8080|:9400|:9401'

   # Kill processes using the ports if needed
   fuser -k 8000/tcp 8080/tcp 9400/tcp 9401/tcp
   ```

3. **Container Issues:**
   ```bash
   # Check container status
   docker ps -a

   # View container logs
   docker logs <container_name>

   # Clean up stuck containers
   docker stop $(docker ps -aq) || true
   docker rm $(docker ps -aq) || true
   ```

4. **Dynamo-Specific Issues:**
   ```bash
   # Check Docker Compose services
   docker compose -f docker-compose.yml ps

   # View Dynamo server logs
   tail -f dynamo_server.log

   # Test Dynamo DCGM endpoint
   curl localhost:9401/metrics | head -20
   ```

5. **Custom DCGM Setup Issues:**
   ```bash
   # Test custom DCGM endpoint (forwarded to port 9401)
   curl localhost:9401/metrics | head -20

   # Check vLLM model loading
   docker logs vllm-server | grep -i "model"

   # Test vLLM endpoint
   curl localhost:8000/v1/models
   ```

6. **Test Framework Issues:**
   ```bash
   # Run with Python debugging
   python3 -u main.py --all-servers

   # Check the log file for detailed output
   cat test_execution.log
   ```

---

# 📚 Additional Resources

- **[NVIDIA DCGM Exporter](https://github.com/NVIDIA/dcgm-exporter)** - Official DCGM Exporter documentation and configuration options
- **[AIPerf Documentation](../../README.md)** - Main AIPerf documentation
- **[DCGM Metrics Reference](https://docs.nvidia.com/datacenter/dcgm/latest/dcgm-api/dcgm-api-field-ids.html)** - Complete list of available GPU metrics

---

# 💡 Quick Reference

## Common Commands

**Basic benchmark with GPU telemetry:**
```bash
aiperf profile --model MODEL_NAME --url localhost:8000 --endpoint-type chat --gpu-telemetry --verbose
```

**Multi-node telemetry collection:**
```bash
aiperf profile --model MODEL_NAME --url localhost:8000 --endpoint-type chat \
  --gpu-telemetry http://node1:9400/metrics http://node2:9400/metrics --verbose
```

**Check if DCGM is running:**
```bash
curl localhost:9401/metrics | grep DCGM_FI_DEV_GPU_UTIL
```

## Key Takeaways

✅ AIPerf always attempts to collect from `http://localhost:9401/metrics` by default
✅ Use `--gpu-telemetry` flag (with `--verbose`) to display GPU metrics in console
✅ Add custom URLs after `--gpu-telemetry` to collect from multiple nodes
✅ Path 2 (custom DCGM) works with **any inference backend** - not just vLLM
✅ Both paths use port 9401, so AIPerf commands are identical
