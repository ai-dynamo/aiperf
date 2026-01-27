<!--
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
-->
# Multi-URL Load Balancing

AIPerf supports distributing requests across multiple inference server instances for horizontal scaling. This is useful for:

- **Multi-GPU scaling**: Run multiple inference containers on a single node, each serving a different GPU
- **Distributed inference**: Load balance across multiple inference servers
- **High-throughput benchmarking**: Aggregate throughput from multiple instances

## Usage

Specify multiple `--url` options to enable load balancing:

```bash
# Round-robin across two servers
aiperf profile --model llama \
    --url http://server1:8000 \
    --url http://server2:8000 \
    --request-rate 20 \
    --request-count 100

# Multi-GPU scaling on a single node
aiperf profile --model llama \
    --url http://localhost:8000 \
    --url http://localhost:8001 \
    --url http://localhost:8002 \
    --url http://localhost:8003 \
    --concurrency 32 \
    --benchmark-duration 60
```

## URL Selection Strategy

Currently supported strategies:

| Strategy | Description |
|----------|-------------|
| `round_robin` (default) | Distributes requests evenly across URLs in sequential order |

You can explicitly set the strategy with `--url-strategy`:

```bash
aiperf profile --model llama \
    --url http://server1:8000 \
    --url http://server2:8000 \
    --url-strategy round_robin \
    --request-count 100
```

## CLI Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--url` | list | localhost:8000 | One or more endpoint URLs; multiple URLs enable load balancing |
| `--url-strategy` | enum | round_robin | Strategy for distributing requests across multiple URLs |

## Behavior Notes

- **Server metrics**: Metrics are collected from all configured URLs
- **Backward compatibility**: Single URL usage remains unchanged
- **Per-request assignment**: Each request is assigned a URL at credit issuance time
- **Connection reuse**: The `--connection-reuse-strategy` applies per-URL
