<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Request Cancellation Testing

AIPerf supports request timeout and cancellation scenarios, which are important for calculating the impact of user cancellation on performance.


## Basic Request Cancellation

Test with a small percentage of cancelled requests:

<!-- aiperf-run-basic-cancellation -->
```bash
# Profile with 10% request cancellation
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --endpoint /v1/chat/completions \
    --streaming \
    --url localhost:8000 \
    --request-cancellation-rate 10 \
    --request-cancellation-delay 0.5 \
    --synthetic-input-tokens-mean 800 \
    --synthetic-input-tokens-stddev 80 \
    --output-tokens-mean 400 \
    --output-tokens-stddev 40 \
    --concurrency 8 \
    --request-count 50 \
    --warmup-request-count 5
```
<!-- /aiperf-run-basic-cancellation -->

**Parameters Explained:**
- `--request-cancellation-rate 10`: Cancel 10% of requests (value between 0.0 and 1.0)
- `--request-cancellation-delay 0.5`: Wait .5 seconds before cancelling selected requests

### High Cancellation Rate Testing

Test service resilience under frequent cancellations:

<!-- aiperf-run-high-cancellation -->
```bash
# Profile with 50% request cancellation
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --endpoint /v1/chat/completions \
    --streaming \
    --url localhost:8000 \
    --request-cancellation-rate 50 \
    --request-cancellation-delay 1.0 \
    --synthetic-input-tokens-mean 1200 \
    --output-tokens-mean 600 \
    --concurrency 10 \
    --request-count 40
```
<!-- /aiperf-run-high-cancellation -->

### Immediate Cancellation Testing

Test rapid cancellation scenarios:

<!-- aiperf-run-immediate-cancellation -->
```bash
# Profile with immediate cancellation (0 delay)
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --endpoint /v1/chat/completions \
    --streaming \
    --url localhost:8000 \
    --request-cancellation-rate 30 \
    --request-cancellation-delay 0.0 \
    --synthetic-input-tokens-mean 500 \
    --output-tokens-mean 100 \
    --concurrency 15 \
    --request-count 60
```
<!-- /aiperf-run-immediate-cancellation -->

**Expected Results:**
- Tests how quickly the server can handle connection terminations
- Useful for testing resource cleanup and connection pooling
