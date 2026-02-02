<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# User Interface

AIPerf provides 3 UI types to display benchmark progress.

## Overview

Choose the UI type that matches your workflow:

- **Dashboard** (default): Full Textual TUI with real-time metrics and GPU telemetry
- **Simple**: TQDM progress bars for minimal overhead
- **None**: Application logs only, no progress UI

All types display the final metrics table upon completion. UI type only affects progress display during benchmark execution.

**Note:** Both `--ui-type` and `--ui` are supported interchangeably.

## Dashboard (Default)

The full-featured TUI provides:
- Real-time request and record metrics
- Live GPU telemetry (when enabled)
- Worker status monitoring
- Interactive display with multiple tabs

```bash
aiperf profile \
  --model Qwen/Qwen3-0.6B \
  --url localhost:8000 \
  --endpoint-type chat \
  --concurrency 10 \
  --request-count 100 \
  --streaming
```

**When to use:**
- Monitoring benchmarks interactively
- Viewing real-time metrics as they're computed
- Checking worker status and errors
- Viewing GPU telemetry

**Note:** Dashboard automatically switches to `simple` when using `--verbose` or `--extra-verbose` for better log visibility.

## Simple

Lightweight progress bars using TQDM:

```bash
aiperf profile \
  --model Qwen/Qwen3-0.6B \
  --url localhost:8000 \
  --endpoint-type chat \
  --concurrency 10 \
  --request-count 100 \
  --streaming \
  --ui-type simple
```

**Sample Output (Successful Run):**
```
INFO     Starting AIPerf System
INFO     Registered Dataset Manager (id: 'dataset_manager_abc123')
INFO     AIPerf System is PROFILING

Profiling: 100/100 |████████████████████████| 100% [01:45<00:00]
Processing Records: 100/100 |███████████████| 100% [00:01<00:00]

INFO     Benchmark completed successfully
INFO     Results saved to: artifacts/Qwen_Qwen3-0.6B-chat-concurrency10/
```

**When to use:**
- Need lower resource overhead
- Terminal doesn't support rich TUI rendering
- Running with verbose logging

## None

Shows application logs only, no progress UI:

```bash
aiperf profile \
  --model Qwen/Qwen3-0.6B \
  --url localhost:8000 \
  --endpoint-type chat \
  --concurrency 10 \
  --request-count 100 \
  --streaming \
  --ui-type none
```

**Sample Output (Successful Run):**
```
23:07:28.809795 INFO     Starting AIPerf System
23:07:31.580751 INFO     Registered Dataset Manager (id: 'dataset_manager_abc123')
23:07:31.586297 INFO     Registered Worker Manager (id: 'worker_manager_def456')
23:07:31.595391 INFO     Registered Timing Manager (id: 'timing_manager_ghi789')
23:07:31.987169 INFO     AIPerf System is CONFIGURING
23:07:32.594891 INFO     AIPerf System is CONFIGURED
23:07:32.597896 INFO     AIPerf System is PROFILING
23:09:18.123456 INFO     Benchmark completed successfully
23:09:18.234567 INFO     Results saved to: artifacts/Qwen_Qwen3-0.6B-chat-concurrency10/
```

**When to use:**
- Automating benchmarks in scripts or CI/CD
- Piping output to files
- Minimizing UI overhead
