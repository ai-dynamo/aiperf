---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: AI Agent Deployment Guide
---

# AI Agent Deployment Guide for AIPerf on Kubernetes

Step-by-step deployment playbook for AI coding agents using `dev/kube.py`. Each step has a command with `--json` output, a machine-checkable gate, and the exact next action. No judgment calls required -- follow the steps in order, verify each gate, and proceed or fix.

All commands use `--json` (structured output to stdout, logs to stderr) and `--yes` (auto-accept confirmations). These flags work before or after the subcommand name.

---

## Overview

```
Step 1: Check Prerequisites (doctor)
Step 2: Setup Cluster + Infrastructure (setup)
Step 3: Deploy Inference Server (deploy-mock, deploy-vllm, or deploy-dynamo)
Step 4: Run Benchmark (run)
Step 5: Monitor Until Completion
Step 6: Collect Results
Step 7: Cleanup (cleanup or teardown)
```

---

## Step 1: Check Prerequisites

```bash
./dev/kube.py doctor --json --yes 2>/dev/null
```

Parse the output:

```python
result = json.loads(output)["result"]
missing = result["missing_required"]
docker_ok = result["docker_running"]
gpu = result["gpu"]
```

**GATE:** `missing_required == []` and `docker_running == true`.

If tools are missing, run `./dev/kube.py doctor --yes` interactively to auto-install them, or install manually:

| Tool | Install (Arch) | Install (Debian) | Install (macOS) |
|---|---|---|---|
| `docker` | `sudo pacman -S docker` | `curl -fsSL https://get.docker.com \| sh` | `brew install --cask docker` |
| `kind` | `sudo pacman -S kind` | See [kind docs](https://kind.sigs.k8s.io/) | `brew install kind` |
| `kubectl` | `sudo pacman -S kubectl` | See [kubectl docs](https://kubernetes.io/docs/tasks/tools/) | `brew install kubectl` |
| `helm` | `sudo pacman -S helm` | See [helm docs](https://helm.sh/docs/intro/install/) | `brew install helm` |

### GPU Detection

```python
has_gpu = gpu["nvidia_smi"]
has_ctk = gpu["nvidia_ctk"]
docker_nvidia = gpu.get("docker_runtime") == "nvidia"
```

For GPU benchmarking, all three must be true. For CPU-only (mock server), GPU is not required.

### Choosing a Cluster Runtime

| Runtime | Default when | GPU support | Set via |
|---|---|---|---|
| Minikube | GPU detected | Yes (`--gpus all`, simpler) | `CLUSTER_RUNTIME=minikube` |
| Kind | No GPU | Yes (nvidia-ctk + device plugin) | `CLUSTER_RUNTIME=kind` |

The runtime auto-selects based on GPU presence. Override with:

```bash
export CLUSTER_RUNTIME=kind      # force Kind even with GPU
export CLUSTER_RUNTIME=minikube  # force Minikube without GPU
```

---

## Step 2: Setup Cluster + Infrastructure

```bash
./dev/kube.py setup --json --yes 2>/dev/null
```

This streams JSONL events -- one per line:

```python
import json, subprocess, sys

proc = subprocess.Popen(
    ["./dev/kube.py", "setup", "--json", "--yes"],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
)

for line in proc.stdout:
    event = json.loads(line)

    if event["type"] == "step_started":
        print(f"Starting: {event['step']}")

    elif event["type"] == "output":
        pass  # subprocess output, useful for debugging

    elif event["type"] == "step_completed":
        print(f"Done: {event['step']} ({event.get('seconds', '?')}s)")

    elif event["type"] == "step_failed":
        print(f"FAILED: {event['step']}: {event['error']}")

    elif event["type"] == "step_skipped":
        print(f"Skipped: {event['step']}: {event['reason']}")

    elif event["type"] == "summary":
        steps = event["result"]["steps"]
        failed = [s for s in steps if s["status"] == "error"]
        if failed:
            print(f"Setup failed: {failed}")
            sys.exit(1)
        print(f"Setup complete in {event['result']['elapsed_seconds']}s")
```

**GATE:** Summary event has `status == "ok"` and no steps with `status == "error"`.

### Setup Flags

| Flag | Effect |
|---|---|
| `--no-dynamo` / `-D` | Skip Dynamo operator install (auto-skipped without GPU) |
| `--no-jobset` / `-J` | Skip JobSet controller install |
| `--no-mock` / `-M` | Skip mock server deploy |
| `-k` / `--continue-on-error` | Continue past failures in optional steps |

---

## Step 3: Deploy Inference Server

Choose one based on your needs:

### 3a. Mock Server (CPU-only, fastest)

Already deployed by setup. Verify:

```bash
./dev/kube.py status --json 2>/dev/null
```

```python
result = json.loads(output)["result"]
mock_ready = result["infra"]["mock_server"]["ready_replicas"]
assert mock_ready is not None and mock_ready > 0
# Endpoint: http://aiperf-mock-server.default.svc.cluster.local:8000
```

### 3b. vLLM Server (GPU required)

```bash
./dev/kube.py deploy-vllm --json --yes --model Qwen/Qwen3-0.6B --gpus 1 2>/dev/null
```

Streams JSONL events during rollout. Parse:

```python
for line in proc.stdout:
    event = json.loads(line)
    if event["type"] == "summary":
        endpoint = event["result"]["endpoint"]
        # e.g. http://vllm-server.vllm-server.svc.cluster.local:8000/v1
```

**GATE:** Summary `status == "ok"` and `result.endpoint` is set.

### 3c. Dynamo Server (GPU required)

```bash
# Aggregated mode (single GPU)
./dev/kube.py deploy-dynamo --json --yes --mode agg --model Qwen/Qwen3-0.6B --gpus 1 2>/dev/null

# Disaggregated on single GPU (prefill + decode share 1 GPU)
./dev/kube.py deploy-dynamo --json --yes --mode disagg-1gpu --model Qwen/Qwen3-0.6B 2>/dev/null
```

**GATE:** Summary `status == "ok"` and `result.endpoint` is set.

### 3d. LoRA Adapters on Dynamo

```bash
./dev/kube.py deploy-lora --yes \
  --name my-adapter \
  --base-model Qwen/Qwen3-0.6B \
  --source hf://org/lora-repo
```

---

## Step 4: Run Benchmark

### 4a. With a config file

```bash
./dev/kube.py run --json --yes --config benchmark.yaml --workers-max 10 2>/dev/null
```

In `--json` mode, `run` auto-detaches and returns the job info:

```python
result = json.loads(last_line)["result"]
job_id = result["job_id"]
namespace = result["namespace"]
```

**GATE:** Summary `status == "ok"` and `job_id` is set.

### 4b. Single-pod benchmark (pass-through CLI args)

```bash
./dev/kube.py run-local --json --yes -- \
  --model mock \
  --endpoint-type chat \
  --request-count 100 \
  --concurrency 10 2>/dev/null
```

### 4c. Dry run (preview manifests only)

```bash
./dev/kube.py dry-run --json --config benchmark.yaml 2>/dev/null
```

Prints the YAML manifests to stdout. In JSON mode, this is raw YAML (not wrapped in JSON).

---

## Step 5: Monitor Until Completion

### 5a. Using `aiperf kube watch` (recommended)

```bash
aiperf kube watch <JOB_ID> --output json --interval 5
```

Each line is a JSON snapshot:

```python
TERMINAL_PHASES = {"Completed", "Failed", "Cancelled"}

for line in proc.stdout:
    snap = json.loads(line)
    phase = snap["phase"]
    health = snap["diagnosis"]["health"]

    if phase in TERMINAL_PHASES:
        proc.terminate()
        break

    for issue in snap["diagnosis"]["issues"]:
        if issue["severity"] == "critical":
            print(f"CRITICAL: {issue['title']}: {issue['suggested_fix']}")
```

### 5b. Using kubectl poll

```bash
kubectl get aiperfjob <JOB_ID> -n <NAMESPACE> \
  -o jsonpath='{.status.phase} {.status.error}'
```

**GATE:** Phase reaches `Completed`, `Failed`, or `Cancelled`.

---

## Step 6: Collect Results

```bash
aiperf kube results <JOB_ID> --output ./artifacts/<JOB_ID>
```

Verify and extract metrics:

```python
metrics_file = Path(f"./artifacts/{job_id}/profile_export_aiperf.json")
assert metrics_file.exists(), "Results not found"

data = json.loads(metrics_file.read_text())
# Each metric is a nested object with: unit, avg, min, max, p50, p99, std, etc.
# Fields with zero values (e.g. error_request_count) are omitted from the export.
metrics = {
    "request_throughput_avg": data["request_throughput"]["avg"],
    "request_latency_p50": data["request_latency"]["p50"],
    "request_latency_p99": data["request_latency"]["p99"],
    "ttft_avg": data["time_to_first_token"]["avg"],
    "output_token_throughput_avg": data["output_token_throughput"]["avg"],
    "request_count": data["request_count"]["avg"],
    "error_request_count": data.get("error_request_count", {}).get("avg", 0),
}
```

**GATE:** `metrics_file.exists()` and `error_request_count == 0`.

---

## Step 7: Cleanup

### Remove benchmark resources only

```bash
./dev/kube.py cleanup --json --yes 2>/dev/null
```

```python
result = json.loads(output)["result"]
# {"deleted_namespaces": [...], "message": "cleanup complete"}
```

### Tear down entire cluster

```bash
./dev/kube.py teardown --json --yes 2>/dev/null
```

### Remove specific servers

```bash
./dev/kube.py remove-mock --yes
./dev/kube.py remove-vllm --yes
./dev/kube.py remove-dynamo --yes
./dev/kube.py remove-lora --yes --name my-adapter
```

---

## Quick Status Check

At any point, check the full cluster state:

```bash
./dev/kube.py status --json 2>/dev/null
```

Returns:

```json
{
  "type": "summary",
  "command": "status",
  "status": "ok",
  "result": {
    "cluster": {"name": "aiperf", "runtime": "kind", "exists": true, "running": true, "connected": true},
    "images": [{"name": "aiperf:local", "built": true, "size_bytes": 809180388}],
    "infra": {
      "jobset": {"ready_replicas": 1},
      "mock_server": {"ready_replicas": 1},
      "gpu": {"count": 1},
      "dynamo_operator": {"installed": true}
    },
    "servers": {
      "vllm": {"ready_replicas": null},
      "dynamo": {"deployed": false},
      "lora": []
    },
    "workloads": {
      "benchmarks": ["my-benchmark-abc123"],
      "namespaces": ["aiperf-bench-abc123"]
    }
  },
  "errors": []
}
```

---

## JSONL Event Protocol

Streaming commands (`setup`, `build`, `deploy-mock`, `deploy-vllm`, `deploy-dynamo`) emit one JSON object per line:

| Event type | Fields | When |
|---|---|---|
| `step_started` | `step`, `timestamp` | Step begins |
| `output` | `step`, `stream` (stdout/stderr), `line` | Subprocess writes a line |
| `step_completed` | `step`, `detail`, `seconds` (optional) | Step succeeds |
| `step_skipped` | `step`, `reason` | Step was skipped |
| `step_failed` | `step`, `error`, `seconds` (optional) | Step failed |
| `summary` | `command`, `status`, `result`, `errors` | Command finished (always last line) |

Non-streaming commands (`status`, `doctor`, `cleanup`) emit a single `summary` line.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `CLUSTER_NAME` | `aiperf` | Cluster name |
| `CLUSTER_RUNTIME` | auto (minikube w/ GPU, kind w/o) | `kind` or `minikube` |
| `AIPERF_IMAGE` | `aiperf:local` | AIPerf container image |
| `CONFIG` | (none) | Benchmark config file |
| `WORKERS` | auto (min of CPUs, 10) | Number of workers |
| `MODEL` | `Qwen/Qwen3-0.6B` | Model name for server deployment |
| `GPUS` | `1` | GPUs per server instance |
| `VLLM_IMAGE` | `vllm/vllm-openai:latest` | vLLM container image |
| `DYNAMO_IMAGE` | `nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.9.0` | Dynamo container image |
| `DYNAMO_MODE` | `agg` | `agg`, `disagg`, or `disagg-1gpu` |
| `HF_TOKEN` | (none) | Hugging Face token for gated models |
| `MAX_MODEL_LEN` | `4096` | Max context length |
| `GPU_MEM_UTIL` | `0.5` | GPU memory utilization |

---

## Complete Agent Workflow Script

Self-contained Python script for the full local pipeline:

```python
#!/usr/bin/env python3
"""AIPerf local K8s benchmark agent using dev/kube.py."""

import json
import subprocess
import sys
import time
from pathlib import Path

KUBE_PY = "./dev/kube.py"


def run_json(cmd: list[str], stream: bool = False) -> dict | list[dict]:
    """Run a command and parse JSON output. Returns summary dict or list of all events."""
    proc = subprocess.run(cmd, capture_output=True, text=True)
    lines = [json.loads(ln) for ln in proc.stdout.strip().splitlines() if ln.strip()]
    if stream:
        return lines
    return lines[-1] if lines else {"status": "error", "errors": ["no output"]}


def run_streaming(cmd: list[str], on_event=None) -> dict:
    """Run a streaming command, calling on_event for each JSONL line. Returns summary."""
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    summary = None
    for line in proc.stdout:
        event = json.loads(line)
        if on_event:
            on_event(event)
        if event["type"] == "summary":
            summary = event
    proc.wait()
    return summary or {"status": "error", "errors": ["no summary"]}


def gate(summary: dict, msg: str) -> None:
    """Assert summary status is ok."""
    if summary.get("status") != "ok":
        print(f"GATE FAILED: {msg}: {summary.get('errors', [])}", file=sys.stderr)
        sys.exit(1)


def log_event(event: dict) -> None:
    t = event["type"]
    if t == "step_started":
        print(f"  -> {event['step']}...")
    elif t == "step_completed":
        print(f"  OK {event['step']} ({event.get('seconds', 0)}s)")
    elif t == "step_failed":
        print(f"  FAIL {event['step']}: {event['error']}")
    elif t == "step_skipped":
        print(f"  -- {event['step']}: {event['reason']}")


def main(
    config_file: str | None = None,
    server: str = "mock",
    model: str = "Qwen/Qwen3-0.6B",
    gpus: int = 1,
    workers: int = 10,
) -> dict:
    # Step 1: Prerequisites
    print("Step 1: Checking prerequisites...")
    doctor = run_json([KUBE_PY, "doctor", "--json", "--yes"])
    gate(doctor, "prerequisites")
    result = doctor["result"]
    if result["missing_required"]:
        print(f"Missing tools: {result['missing_required']}", file=sys.stderr)
        sys.exit(1)

    # Step 2: Setup
    print("Step 2: Setting up cluster...")
    summary = run_streaming([KUBE_PY, "setup", "--json", "--yes"], on_event=log_event)
    gate(summary, "setup")

    # Step 3: Deploy server
    print(f"Step 3: Deploying {server} server...")
    if server == "mock":
        status = run_json([KUBE_PY, "status", "--json"])
        mock_ready = status["result"]["infra"]["mock_server"]["ready_replicas"]
        if not mock_ready:
            print("Mock server not ready after setup", file=sys.stderr)
            sys.exit(1)
    elif server == "vllm":
        summary = run_streaming(
            [KUBE_PY, "deploy-vllm", "--json", "--yes", "--model", model, "--gpus", str(gpus)],
            on_event=log_event,
        )
        gate(summary, "deploy-vllm")
    elif server == "dynamo":
        summary = run_streaming(
            [KUBE_PY, "deploy-dynamo", "--json", "--yes", "--mode", "agg", "--model", model, "--gpus", str(gpus)],
            on_event=log_event,
        )
        gate(summary, "deploy-dynamo")

    # Step 4: Run benchmark
    print("Step 4: Running benchmark...")
    run_args = [KUBE_PY, "run", "--json", "--yes", "--workers-max", str(workers)]
    if config_file:
        run_args.extend(["--config", config_file])
    summary = run_json(run_args)
    gate(summary, "run")
    job_id = summary["result"]["job_id"]
    namespace = summary["result"]["namespace"]
    print(f"  Job: {job_id} in {namespace}")

    # Step 5: Monitor
    print("Step 5: Monitoring...")
    proc = subprocess.Popen(
        ["aiperf", "kube", "watch", job_id, "--output", "json", "--interval", "5"],
        stdout=subprocess.PIPE, text=True,
    )
    final_phase = "Unknown"
    try:
        for line in proc.stdout:
            snap = json.loads(line)
            final_phase = snap["phase"]
            health = snap["diagnosis"]["health"]
            pct = snap.get("progress", {}).get("percent", 0)
            print(f"  Phase: {final_phase}, Health: {health}, Progress: {pct:.0f}%")
            if final_phase in {"Completed", "Failed", "Cancelled"}:
                break
    finally:
        proc.terminate()

    if final_phase != "Completed":
        return {"success": False, "error": f"Benchmark ended: {final_phase}"}

    # Step 6: Collect results
    print("Step 6: Collecting results...")
    artifacts = f"./artifacts/{job_id}"
    subprocess.run(["aiperf", "kube", "results", job_id, "--output", artifacts], check=False)
    metrics_file = Path(artifacts) / "profile_export_aiperf.json"
    metrics = json.loads(metrics_file.read_text()) if metrics_file.exists() else {}

    return {"success": True, "job_id": job_id, "metrics": metrics, "artifacts_dir": artifacts}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Benchmark config YAML")
    parser.add_argument("--server", default="mock", choices=["mock", "vllm", "dynamo"])
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    result = main(
        config_file=args.config,
        server=args.server,
        model=args.model,
        gpus=args.gpus,
        workers=args.workers,
    )
    print(json.dumps(result, indent=2, default=str))
    sys.exit(0 if result["success"] else 1)
```

Usage:

```bash
# CPU-only with mock server (fastest, no GPU needed)
python3 deploy_agent.py --server mock --config benchmark.yaml

# vLLM with real GPU
python3 deploy_agent.py --server vllm --model Qwen/Qwen3-0.6B --gpus 1

# Dynamo with Minikube GPU passthrough
CLUSTER_RUNTIME=minikube python3 deploy_agent.py --server dynamo --model Qwen/Qwen3-0.6B
```

---

## GPU Cluster Details

GPU systems default to Minikube (`--gpus all`). Setup auto-detects and configures GPU passthrough:

```bash
# GPU auto-detected, minikube selected automatically
./dev/kube.py setup --json --yes 2>/dev/null
```

Requirements:
- `nvidia-smi` on host (NVIDIA drivers installed)
- Docker with NVIDIA Container Toolkit
- Minikube uses `--driver docker --container-runtime docker --gpus all`

Verify GPU in cluster:

```python
status = run_json(["./dev/kube.py", "status", "--json"])
gpu_count = status["result"]["infra"]["gpu"]["count"]
assert gpu_count > 0, f"Expected GPU, got {gpu_count}"
```

To force Kind instead (requires nvidia-ctk + device plugin inside the node):

```bash
CLUSTER_RUNTIME=kind ./dev/kube.py setup --json --yes 2>/dev/null
```

---

## Exit Codes

| Code | Meaning |
|---|---|
| 0 | Success |
| 1 | Failure (any step) |

---

## Related Documentation

- [AI Agent Debugging Guide](ai-debugging-guide.md) -- Diagnosing and fixing benchmark issues
- [Getting Started](getting-started.md) -- Human-readable getting started guide
- [Kubernetes Configuration](configuration.md) -- Complete CRD and CLI reference
