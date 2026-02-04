<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Fixed Schedule Benchmarking

Fixed schedule benchmarking provides precise timing control by executing requests at specific timestamps.
This mode is ideal for simulating exact traffic patterns, testing temporal performance characteristics,
and reproducing time-sensitive scenarios.

## Overview

Fixed schedule mode enables:

- **Precise Timing**: Execute requests at exact millisecond intervals
- **Traffic Simulation**: Replicate real-world traffic patterns
- **Performance Analysis**: Identify how response times vary with request timing
- **Load Testing**: Test system behavior under controlled temporal stress patterns

## Fixed Schedule File Format

Fixed schedule files use JSONL format with timestamp-based entries:

```jsonl
{"timestamp": 0, "input_length": 100, "output_length": 200, "hash_ids": [1001]}
{"timestamp": 500, "input_length": 200, "output_length": 400, "hash_ids": [1002]}
{"timestamp": 1000, "input_length": 550, "output_length": 500, "hash_ids": [1003, 1005]}
```

**Field Descriptions:**
- `timestamp`: Milliseconds from schedule start when request should be sent
- `input_length`: Number of tokens in the input prompt
- `input_text`: Exact text to send in the request (provided instead of input_length)
- `output_length`: Maximum number of tokens in the response (optional)
- `hash_ids`: Hash block identifiers to simulate text reuse with 512-token blocks (optional)

## Basic Fixed Schedule Execution

### Setting Up the Server

```bash
# Start vLLM server for fixed schedule testing
docker pull vllm/vllm-openai:latest
docker run --gpus all -p 8000:8000 vllm/vllm-openai:latest \
  --model Qwen/Qwen3-0.6B \
  --host 0.0.0.0 --port 8000 &
```

```bash
# Wait for server to be ready
timeout 900 bash -c 'while [ "$(curl -s -o /dev/null -w "%{http_code}" localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"Qwen/Qwen3-0.6B\",\"messages\":[{\"role\":\"user\",\"content\":\"test\"}],\"max_tokens\":1}")" != "200" ]; do sleep 2; done' || { echo "vLLM not ready after 15min"; exit 1; }
```

### Running Basic Fixed Schedule

<!-- aiperf-run-vllm-default-openai-endpoint-server -->
```bash
# Create a fixed schedule with precise timing
cat > precise_schedule.jsonl << 'EOF'
{"timestamp": 0, "input_length": 100, "hash_ids": [3001]}
{"timestamp": 500, "input_length": 200, "hash_ids": [3002]}
{"timestamp": 750, "input_length": 150, "hash_ids": [3003]}
{"timestamp": 1000, "input_length": 300, "hash_ids": [3004]}
{"timestamp": 1250, "input_length": 180, "hash_ids": [3005]}
{"timestamp": 2000, "input_length": 400, "hash_ids": [3006]}
{"timestamp": 2500, "input_length": 250, "hash_ids": [3007]}
{"timestamp": 3000, "input_length": 350, "hash_ids": [3008]}
{"timestamp": 4000, "input_length": 500, "hash_ids": [3009]}
{"timestamp": 5000, "input_length": 600, "hash_ids": [3010, 3050]}
EOF
# Run basic fixed schedule benchmarking
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --endpoint /v1/chat/completions \
    --streaming \
    --url localhost:8000 \
    --input-file precise_schedule.jsonl \
    --custom-dataset-type mooncake_trace \
    --fixed-schedule \
    --fixed-schedule-auto-offset
```
<!-- /aiperf-run-vllm-default-openai-endpoint-server -->

**Sample Output (Successful Run):**
```
INFO     Starting AIPerf System
INFO     Using Fixed Schedule mode with auto-offset
INFO     Loaded 10 entries from precise_schedule.jsonl
INFO     Schedule duration: 5.0 seconds
INFO     AIPerf System is PROFILING

Profiling: 10/10 |████████████████████████| 100% [00:05<00:00]

INFO     Benchmark completed successfully
INFO     Results saved to: artifacts/Qwen_Qwen3-0.6B-chat-fixed-schedule/

            NVIDIA AIPerf | LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┓
┃                     Metric ┃    avg ┃    min ┃    max ┃    p99 ┃    p50 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━┩
│       Request Latency (ms) │ 345.67 │ 234.56 │ 498.12 │ 476.34 │ 338.90 │
│   Time to First Token (ms) │  78.45 │  52.34 │ 112.67 │ 108.23 │  76.12 │
│   Inter Token Latency (ms) │  15.23 │  11.45 │  22.34 │  21.12 │  14.89 │
│ Request Throughput (req/s) │   2.89 │      - │      - │      - │      - │
└────────────────────────────┴────────┴────────┴────────┴────────┴────────┘

JSON Export: artifacts/Qwen_Qwen3-0.6B-chat-fixed-schedule/profile_export_aiperf.json
```

**Key Parameters:**
- `--fixed-schedule-auto-offset`: Automatically adjusts timestamps to start from 0

## Advanced Schedule Patterns

### Time Window Execution

Execute only a portion of the schedule using start and end offsets:

<!-- aiperf-run-vllm-default-openai-endpoint-server -->
```bash
# Execute schedule from 2s to 6s window
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --endpoint /v1/chat/completions \
    --streaming \
    --url localhost:8000 \
    --input-file precise_schedule.jsonl \
    --custom-dataset-type mooncake_trace \
    --fixed-schedule \
    --fixed-schedule-start-offset 2000 \
    --fixed-schedule-end-offset 4000
```
<!-- /aiperf-run-vllm-default-openai-endpoint-server -->

**Sample Output (Successful Run):**
```
INFO     Starting AIPerf System
INFO     Using Fixed Schedule mode with time window [2000ms - 4000ms]
INFO     Loaded 10 entries from precise_schedule.jsonl
INFO     Filtered to 2 entries within time window
INFO     Schedule duration: 2.0 seconds
INFO     AIPerf System is PROFILING

Profiling: 2/2 |████████████████████████| 100% [00:02<00:00]

INFO     Benchmark completed successfully
INFO     Results saved to: artifacts/Qwen_Qwen3-0.6B-chat-fixed-schedule/

            NVIDIA AIPerf | LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┓
┃                     Metric ┃    avg ┃    min ┃    max ┃    p99 ┃    p50 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━┩
│       Request Latency (ms) │ 389.45 │ 312.67 │ 466.23 │ 466.23 │ 389.45 │
│   Time to First Token (ms) │  89.12 │  71.34 │ 106.90 │ 106.90 │  89.12 │
│   Inter Token Latency (ms) │  16.78 │  14.23 │  19.34 │  19.34 │  16.78 │
│ Request Throughput (req/s) │   1.45 │      - │      - │      - │      - │
└────────────────────────────┴────────┴────────┴────────┴────────┴────────┘

JSON Export: artifacts/Qwen_Qwen3-0.6B-chat-fixed-schedule/profile_export_aiperf.json
```

**Windowing Parameters:**
- `--fixed-schedule-start-offset 2000`: Start execution at 2000ms timestamp
- `--fixed-schedule-end-offset 4000`: End execution at 4000ms timestamp


## Use Cases

> [!IMPORTANT]
> **When to Use Fixed Schedule Benchmarking:**
> - **Traffic Replay**: Reproduce exact timing patterns from production logs
> - **Temporal Analysis**: Study how performance varies with request timing
> - **Peak Load Testing**: Test system behavior during known high-traffic periods
> - **SLA Validation**: Verify performance under specific timing constraints
> - **Capacity Planning**: Model future load based on projected growth patterns
> - **Regression Testing**: Ensure temporal performance characteristics remain stable

## Related Tutorials

- [Custom Prompt Benchmarking](custom-prompt-benchmarking.md) - For sending custom prompts without timing control
- [Time-based Benchmarking](time-based-benchmarking.md) - For duration-based testing
- [Request Cancellation](request-cancellation.md) - For timeout testing