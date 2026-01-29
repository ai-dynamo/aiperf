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

**Example output:**
```
Profiling: 3/5 |████████████████        | 60% [00:03<00:02]
Processing Records: 5/5 |████████████████████| 100% [00:04<00:00]
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

**When to use:**
- Automating benchmarks in scripts or CI/CD
- Piping output to files
- Minimizing UI overhead
