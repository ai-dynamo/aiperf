<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Profile Audio Language Models with AIPerf

AIPerf supports benchmarking Audio Language Models that process audio inputs with optional text prompts.

This guide covers profiling audio models using OpenAI-compatible chat completions endpoints with vLLM.

---

## Start a vLLM Server

Launch the vLLM server with Qwen2-Audio-7B-Instruct. Audio support requires the `vllm[audio]` extras to be installed:

<!-- setup-vllm-audio-openai-endpoint-server -->
```bash
# Build vLLM image with audio support
docker build -t vllm-audio - << 'EOF'
FROM vllm/vllm-openai:latest
RUN pip install 'vllm[audio]'
EOF

# Run the server
docker run --gpus all -p 8000:8000 vllm-audio \
  --model Qwen/Qwen2-Audio-7B-Instruct \
  --trust-remote-code
```
<!-- /setup-vllm-audio-openai-endpoint-server -->


Verify the server is ready:

<!-- health-check-vllm-audio-openai-endpoint-server -->
```bash
timeout 900 bash -c 'while [ "$(curl -s -o /dev/null -w "%{http_code}" localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"Qwen/Qwen2-Audio-7B-Instruct\",\"messages\":[{\"role\":\"user\",\"content\":\"test\"}],\"max_tokens\":1}")" != "200" ]; do sleep 2; done' || { echo "vLLM not ready after 15min"; exit 1; }
```
<!-- /health-check-vllm-audio-openai-endpoint-server -->

---

## Profile with Synthetic Audio

AIPerf can generate synthetic audio for benchmarking:

<!-- aiperf-run-vllm-audio-openai-endpoint-server -->
```bash
aiperf profile \
    --model Qwen/Qwen2-Audio-7B-Instruct \
    --endpoint-type chat \
    --audio-length-mean 5.0 \
    --audio-format wav \
    --audio-sample-rates 16 \
    --streaming \
    --url localhost:8000 \
    --request-count 20 \
    --concurrency 4
```
<!-- /aiperf-run-vllm-audio-openai-endpoint-server -->

**Output:**

```

                                            NVIDIA AIPerf | LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━┓
┃                               Metric ┃      avg ┃    min ┃       max ┃       p99 ┃       p90 ┃    p50 ┃      std ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━┩
│             Time to First Token (ms) │ 3,658.78 │ 191.80 │ 17,055.13 │ 17,050.10 │ 17,028.62 │ 354.35 │ 6,688.15 │
│            Time to Second Token (ms) │    56.19 │   6.48 │    180.49 │    179.90 │    102.05 │  25.66 │    49.92 │
│      Time to First Output Token (ms) │ 3,658.78 │ 191.80 │ 17,055.13 │ 17,050.10 │ 17,028.62 │ 354.35 │ 6,688.15 │
│                 Request Latency (ms) │ 4,168.43 │ 315.29 │ 17,786.34 │ 17,721.50 │ 17,422.68 │ 841.08 │ 6,658.54 │
│             Inter Token Latency (ms) │    39.17 │  24.35 │     76.16 │     72.60 │     56.47 │  35.58 │    13.24 │
│     Output Token Throughput Per User │    28.17 │  13.13 │     41.06 │     41.04 │     40.83 │  28.10 │     8.31 │
│                    (tokens/sec/user) │          │        │           │           │           │        │          │
│      Output Sequence Length (tokens) │    14.85 │   5.00 │     74.00 │     64.12 │     19.30 │  12.00 │    14.35 │
│       Input Sequence Length (tokens) │   550.00 │ 550.00 │    550.00 │    550.00 │    550.00 │ 550.00 │     0.00 │
│ Output Token Throughput (tokens/sec) │    13.62 │    N/A │       N/A │       N/A │       N/A │    N/A │      N/A │
│    Request Throughput (requests/sec) │     0.92 │    N/A │       N/A │       N/A │       N/A │    N/A │      N/A │
│             Request Count (requests) │    20.00 │    N/A │       N/A │       N/A │       N/A │    N/A │      N/A │
└──────────────────────────────────────┴──────────┴────────┴───────────┴───────────┴───────────┴────────┴──────────┘

CLI Command: aiperf profile --model 'Qwen/Qwen2-Audio-7B-Instruct' --endpoint-type 'chat' --audio-length-mean 5.0
--audio-format 'wav' --audio-sample-rates 16 --streaming --url 'localhost:8000' --request-count 20 --concurrency 4
Benchmark Duration: 21.80 sec
CSV Export:
/home/lkomali/aiperf/artifacts/Qwen_Qwen2-Audio-7B-Instruct-openai-chat-concurrency4/profile_export_aiperf.csv
JSON Export:
/home/lkomali/aiperf/artifacts/Qwen_Qwen2-Audio-7B-Instruct-openai-chat-concurrency4/profile_export_aiperf.json
Log File: /home/lkomali/aiperf/artifacts/Qwen_Qwen2-Audio-7B-Instruct-openai-chat-concurrency4/logs/aiperf.log
```

To add text prompts alongside audio, include `--synthetic-input-tokens-mean 100`

## Profile with Custom Input File

AIPerf can automatically load and encode audio files from local paths.

> **Note:** The example below uses paths from the AIPerf test fixtures directory. Replace these with paths to your own audio files.

<!-- aiperf-run-vllm-audio-openai-endpoint-server -->
```bash
cat <<EOF > inputs.jsonl
{"texts": ["Transcribe this."], "audios": ["/fixtures/audio/test_audio_1s.wav"]}
{"texts": ["What is said?"], "audios": ["/fixtures/audio/test_audio_2.wav"]}
{"texts": ["Summarize."], "audios": ["/fixtures/audio/test_audio_3.wav"]}
EOF

aiperf profile \
    --model Qwen/Qwen2-Audio-7B-Instruct \
    --endpoint-type chat \
    --input-file inputs.jsonl \
    --custom-dataset-type single_turn \
    --streaming \
    --url localhost:8000 \
    --request-count 3
```
<!-- /aiperf-run-vllm-audio-openai-endpoint-server -->

AIPerf will automatically:
- Load the audio files from the specified paths
- Convert them to base64 format
- Send them to the model endpoint

**Output:**

```

                                          NVIDIA AIPerf | LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┓
┃                               Metric ┃      avg ┃    min ┃      max ┃      p99 ┃      p90 ┃    p50 ┃    std ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━┩
│             Time to First Token (ms) │   102.36 │  85.26 │   135.83 │   134.83 │   125.86 │  85.99 │  23.67 │
│            Time to Second Token (ms) │    21.98 │  21.57 │    22.48 │    22.47 │    22.36 │  21.90 │   0.38 │
│      Time to First Output Token (ms) │   102.36 │  85.26 │   135.83 │   134.83 │   125.86 │  85.99 │  23.67 │
│                 Request Latency (ms) │ 1,036.43 │ 433.65 │ 2,127.44 │ 2,095.85 │ 1,811.59 │ 548.20 │ 772.87 │
│             Inter Token Latency (ms) │    21.72 │  21.70 │    21.73 │    21.73 │    21.73 │  21.73 │   0.01 │
│     Output Token Throughput Per User │    46.04 │  46.02 │    46.08 │    46.07 │    46.07 │  46.03 │   0.02 │
│                    (tokens/sec/user) │          │        │          │          │          │        │        │
│      Output Sequence Length (tokens) │    44.00 │  17.00 │    95.00 │    93.50 │    80.00 │  20.00 │  36.08 │
│       Input Sequence Length (tokens) │     4.00 │   4.00 │     4.00 │     4.00 │     4.00 │   4.00 │   0.00 │
│ Output Token Throughput (tokens/sec) │    41.81 │    N/A │      N/A │      N/A │      N/A │    N/A │    N/A │
│    Request Throughput (requests/sec) │     0.95 │    N/A │      N/A │      N/A │      N/A │    N/A │    N/A │
│             Request Count (requests) │     3.00 │    N/A │      N/A │      N/A │      N/A │    N/A │    N/A │
└──────────────────────────────────────┴──────────┴────────┴──────────┴──────────┴──────────┴────────┴────────┘

CLI Command: aiperf profile --model 'Qwen/Qwen2-Audio-7B-Instruct' --endpoint-type 'chat' --input-file
'inputs_filepaths.jsonl' --custom-dataset-type 'single_turn' --streaming --url 'localhost:8000' --request-count 3
Benchmark Duration: 3.16 sec
CSV Export:
/home/lkomali/aiperf/artifacts/Qwen_Qwen2-Audio-7B-Instruct-openai-chat-concurrency1/profile_export_aiperf.csv
JSON Export:
/home/lkomali/aiperf/artifacts/Qwen_Qwen2-Audio-7B-Instruct-openai-chat-concurrency1/profile_export_aiperf.json
Log File: /home/lkomali/aiperf/artifacts/Qwen_Qwen2-Audio-7B-Instruct-openai-chat-concurrency1/logs/aiperf.log
```