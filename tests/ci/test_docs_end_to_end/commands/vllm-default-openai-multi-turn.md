# CI commands for multi-turn doc tests (vllm-default-openai server)
#
# These commands mirror the examples in docs/tutorials/multi-turn.mdx.
# Kept here (outside the docs tree) so CI markers never appear in rendered docs.

{/* aiperf-run-vllm-default-openai-endpoint-server */}
```bash
# Fixed-length conversations: 10 conversations x 3 turns
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --endpoint /v1/chat/completions \
    --streaming \
    --url localhost:8000 \
    --conversation-num 10 \
    --conversation-turn-mean 3 \
    --conversation-turn-stddev 0 \
    --synthetic-input-tokens-mean 200 \
    --output-tokens-mean 150 \
    --concurrency 2 \
    --random-seed 42
```
{/* /aiperf-run-vllm-default-openai-endpoint-server */}

{/* aiperf-run-vllm-default-openai-endpoint-server */}
```bash
# Variable-length conversations: 20 conversations, mean 5 turns
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --endpoint /v1/chat/completions \
    --streaming \
    --url localhost:8000 \
    --conversation-num 20 \
    --conversation-turn-mean 5 \
    --conversation-turn-stddev 2 \
    --synthetic-input-tokens-mean 150 \
    --output-tokens-mean 100 \
    --concurrency 4 \
    --random-seed 42
```
{/* /aiperf-run-vllm-default-openai-endpoint-server */}

{/* aiperf-run-vllm-default-openai-endpoint-server */}
```bash
# Turn delays: 15 conversations with 2000ms mean delay between turns
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --endpoint /v1/chat/completions \
    --streaming \
    --url localhost:8000 \
    --conversation-num 15 \
    --conversation-turn-mean 4 \
    --conversation-turn-stddev 1 \
    --conversation-turn-delay-mean 2000 \
    --conversation-turn-delay-stddev 500 \
    --synthetic-input-tokens-mean 180 \
    --output-tokens-mean 120 \
    --concurrency 3 \
    --random-seed 42
```
{/* /aiperf-run-vllm-default-openai-endpoint-server */}

{/* aiperf-run-vllm-default-openai-endpoint-server */}
```bash
# High-concurrency: 100 conversations, concurrency 50
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --endpoint /v1/chat/completions \
    --streaming \
    --url localhost:8000 \
    --conversation-num 100 \
    --conversation-turn-mean 6 \
    --conversation-turn-stddev 2 \
    --synthetic-input-tokens-mean 250 \
    --output-tokens-mean 200 \
    --concurrency 50 \
    --random-seed 42
```
{/* /aiperf-run-vllm-default-openai-endpoint-server */}

{/* aiperf-run-vllm-default-openai-endpoint-server */}
```bash
# Request rate: 30 conversations at 5 conversations/second (Poisson)
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --endpoint /v1/chat/completions \
    --streaming \
    --url localhost:8000 \
    --conversation-num 30 \
    --conversation-turn-mean 4 \
    --request-rate 5 \
    --request-rate-mode poisson \
    --synthetic-input-tokens-mean 200 \
    --output-tokens-mean 150 \
    --random-seed 42
```
{/* /aiperf-run-vllm-default-openai-endpoint-server */}

{/* aiperf-run-vllm-default-openai-endpoint-server */}
```bash
# Customer support: 50 conversations, 7 turns mean, 3000ms delays
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --endpoint /v1/chat/completions \
    --streaming \
    --url localhost:8000 \
    --conversation-num 50 \
    --conversation-turn-mean 7 \
    --conversation-turn-stddev 2 \
    --conversation-turn-delay-mean 3000 \
    --conversation-turn-delay-stddev 1000 \
    --synthetic-input-tokens-mean 150 \
    --synthetic-input-tokens-stddev 50 \
    --output-tokens-mean 200 \
    --output-tokens-stddev 80 \
    --concurrency 10 \
    --random-seed 42
```
{/* /aiperf-run-vllm-default-openai-endpoint-server */}

{/* aiperf-run-vllm-default-openai-endpoint-server */}
```bash
# Context window stress: 10 conversations x 15 turns
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --endpoint /v1/chat/completions \
    --streaming \
    --url localhost:8000 \
    --conversation-num 10 \
    --conversation-turn-mean 15 \
    --conversation-turn-stddev 3 \
    --synthetic-input-tokens-mean 300 \
    --output-tokens-mean 250 \
    --concurrency 2 \
    --random-seed 42
```
{/* /aiperf-run-vllm-default-openai-endpoint-server */}

{/* aiperf-run-vllm-default-openai-endpoint-server */}
```bash
# Burst traffic: 100 conversations, concurrency 50
aiperf profile \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --endpoint /v1/chat/completions \
    --streaming \
    --url localhost:8000 \
    --conversation-num 100 \
    --conversation-turn-mean 3 \
    --concurrency 50 \
    --synthetic-input-tokens-mean 180 \
    --output-tokens-mean 120 \
    --random-seed 42
```
{/* /aiperf-run-vllm-default-openai-endpoint-server */}
