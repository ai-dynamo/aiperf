<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Request Rate with Max Concurrency

Combining `--request-rate` with `--concurrency` gives you precise control over both request timing *and* the maximum number of concurrent connections. This dual-control approach is essential for:

- **Avoiding thundering herd** — prevents simultaneous request bursts that overwhelm servers
- **Testing real-world API constraints** — validates behavior under actual rate and concurrency limits
- **Simulating realistic clients** — models bandwidth constraints and connection pool limits
- **Validating resource protection** — ensures servers handle properly-constrained traffic
- **Controlled capacity testing** — scales load gradually without resource exhaustion

## How It Works

When both parameters are specified, AIPerf uses a **sleep-then-gate** pattern for each request:

1. **Sleep** — wait according to request rate timing (Poisson or constant intervals)
2. **Acquire semaphore** — block if currently at max concurrency
3. **Issue request** — send to server
4. **On completion** — release semaphore slot for next request

The sleep controls **when** requests attempt to launch (the rate), while the semaphore controls **whether** they can proceed (the concurrency ceiling).

> [!IMPORTANT]
> **No catch-up behavior**: When blocked at max concurrency, the inter-arrival schedule pauses. The system does not attempt to "catch up" by issuing requests faster once slots free up. Hitting the concurrency limit proportionally delays the entire schedule.

> [!TIP]
> **Sustaining max concurrency**: If your request rate is faster than your server's average response time, the system will naturally reach and sustain max concurrency. For example, at 100 req/s with 1 second average response time, new requests arrive every 10ms but each takes 1 second to complete—so slots are always waiting to be filled, keeping the system at the concurrency ceiling. Conversely, if the request rate is slower than response time, slots free up faster than new requests arrive, so concurrency may never reach the maximum.

> [!NOTE]
> **Ramp-up time formula**: `ramp_up_time = concurrency / request_rate`
>
> | Concurrency | Request Rate | Ramp-up Time |
> |-------------|--------------|--------------|
> | 100         | 200 req/s    | ~0.5 seconds |
> | 50          | 50 req/s     | ~1.0 second  |
> | 100         | 20 req/s     | ~5.0 seconds |

## Choosing Your Request Rate Mode

The sleep intervals in step 1 are determined by your chosen request rate mode. AIPerf supports two distribution patterns:

> [!TIP]
> **`poisson` (default)** — Uses exponentially-distributed inter-arrival times to mimic natural traffic with randomized spacing. Ideal for realistic load testing and capacity planning. Add `--random-seed` for reproducible random patterns.
>
> **`constant`** — Requests arrive at precisely evenly-spaced intervals for deterministic, predictable load. Ideal for reproducible benchmarks and regression testing.

## Setting Up the Server

```bash
# Start vLLM server
docker pull vllm/vllm-openai:latest
docker run --gpus all -p 8000:8000 vllm/vllm-openai:latest \
  --model Qwen/Qwen3-0.6B \
  --host 0.0.0.0 --port 8000 &
```

```bash
# Wait for server to be ready
timeout 900 bash -c 'while [ "$(curl -s -o /dev/null -w "%{http_code}" localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"Qwen/Qwen3-0.6B\",\"messages\":[{\"role\":\"user\",\"content\":\"test\"}],\"max_tokens\":1}")" != "200" ]; do sleep 2; done' || { echo "vLLM not ready after 15min"; exit 1; }
```

## Running the Examples

The following examples demonstrate both request rate modes in action. Make sure you've set up the server first (see above).

### Poisson Mode: Natural Traffic Patterns

This example demonstrates realistic traffic simulation with a fast ramp-up (0.5 seconds). The Poisson distribution creates natural variance in request timing while maintaining an average rate of 200 req/s, capped at 100 concurrent requests.

<!-- aiperf-run-vllm-default-openai-endpoint-server -->
```bash
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --endpoint /v1/chat/completions \
    --streaming \
    --url localhost:8000 \
    --concurrency 100 \
    --request-rate 200 \
    --request-rate-mode poisson \
    --random-seed 42 \
    --request-count 500 \
    --synthetic-input-tokens-mean 500 \
    --output-tokens-mean 200
```
<!-- /aiperf-run-vllm-default-openai-endpoint-server -->

### Constant Mode: Deterministic Timing

This example uses evenly-spaced requests with a moderate ramp-up (1 second). Requests arrive at exactly 20ms intervals (50 req/s), making results highly reproducible for benchmarking and regression testing.

<!-- aiperf-run-vllm-default-openai-endpoint-server -->
```bash
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --endpoint /v1/chat/completions \
    --streaming \
    --url localhost:8000 \
    --concurrency 50 \
    --request-rate 50 \
    --request-rate-mode constant \
    --request-count 300 \
    --synthetic-input-tokens-mean 800 \
    --output-tokens-mean 400
```
<!-- /aiperf-run-vllm-default-openai-endpoint-server -->

## Common Use Cases

Here are practical scenarios where combining request rate with max concurrency is particularly valuable:

### High-Concurrency Testing at Scale

Test how your server handles thousands of concurrent users with a controlled ramp-up to avoid overwhelming connection establishment. A longer ramp-up gives the server time to allocate resources gradually.

> **Recommended mode**: Either `poisson` (for realistic variance) or `constant` (for predictable ramp-up)

### API Rate Limit Validation

Validate server behavior when clients respect both throughput and concurrency constraints. This tests rate-limiting logic and ensures proper 429/503 responses when appropriate.

> **Recommended mode**: `constant` for precise, reproducible rate testing

### Realistic User Traffic Simulation

Simulate organic user behavior with natural variance in request timing. The Poisson distribution models real-world patterns where users don't arrive at perfectly regular intervals.

> **Recommended mode**: `poisson` with `--random-seed` for reproducible realistic traffic

## Quick Reference

**Key Parameters:**
- `--request-rate <number>` — Target requests per second
- `--concurrency <number>` — Maximum concurrent requests (acts as ceiling)
- `--request-rate-mode <poisson|constant>` — Request timing distribution (default: `poisson`)
- `--random-seed <number>` — Makes Poisson mode reproducible

**Behavior:**
- Sleep-then-gate pattern: sleep based on rate, then wait for semaphore slot (max concurrency)
- No catch-up: blocked requests pause the schedule
- Sustained concurrency: achieved when request rate exceeds server response time

**Choosing a mode:**
- Use `poisson` for realistic traffic simulation with natural variance
- Use `constant` for reproducible benchmarks with predictable timing
- Add `--random-seed` to `poisson` for reproducible randomness
