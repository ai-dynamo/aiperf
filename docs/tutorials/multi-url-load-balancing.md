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

**Sample Output (Successful Run):**
```
INFO     Starting AIPerf System
INFO     Load balancing enabled: 2 URLs with round_robin strategy
INFO     Using Request_Rate strategy (20.0 req/s)
INFO     AIPerf System is PROFILING

Profiling: 100/100 |████████████████████████| 100% [00:05<00:00]

INFO     Benchmark completed successfully
INFO     Results saved to: artifacts/llama-chat-rate20/

            NVIDIA AIPerf | LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┓
┃                     Metric ┃    avg ┃    min ┃    max ┃    p99 ┃    p50 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━┩
│       Request Latency (ms) │ 234.56 │ 189.34 │ 312.45 │ 298.67 │ 231.23 │
│   Time to First Token (ms) │  56.78 │  45.12 │  78.90 │  75.34 │  55.67 │
│ Request Throughput (req/s) │  19.45 │      - │      - │      - │      - │
└────────────────────────────┴────────┴────────┴────────┴────────┴────────┘

JSON Export: artifacts/llama-chat-rate20/profile_export_aiperf.json
```

# Multi-GPU scaling on a single node
aiperf profile --model llama \
    --url http://localhost:8000 \
    --url http://localhost:8001 \
    --url http://localhost:8002 \
    --url http://localhost:8003 \
    --concurrency 32 \
    --benchmark-duration 60

**Sample Output (Successful Run):**
```
INFO     Starting AIPerf System
INFO     Load balancing enabled: 4 URLs with round_robin strategy
INFO     AIPerf System is PROFILING

Profiling: [01:00] - Running for 60 seconds...

INFO     Benchmark completed successfully
INFO     Results saved to: artifacts/llama-chat-concurrency32/

            NVIDIA AIPerf | LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┓
┃                     Metric ┃    avg ┃    min ┃    max ┃    p99 ┃    p50 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━┩
│       Request Latency (ms) │ 198.34 │ 145.67 │ 289.12 │ 267.45 │ 194.23 │
│   Time to First Token (ms) │  48.90 │  37.23 │  69.45 │  65.78 │  47.89 │
│ Request Throughput (req/s) │  78.90 │      - │      - │      - │      - │
└────────────────────────────┴────────┴────────┴────────┴────────┴────────┘

JSON Export: artifacts/llama-chat-concurrency32/profile_export_aiperf.json
```
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
