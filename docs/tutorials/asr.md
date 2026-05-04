---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Profile ASR Models with Public Datasets
---

# Profile ASR Models with Public Datasets

AIPerf supports benchmarking Automatic Speech Recognition (ASR) models using publicly available
speech datasets from HuggingFace. Each dataset entry sends real speech audio alongside a fixed
"Transcribe this audio." prompt to measure end-to-end transcription latency and throughput.

Five ASR datasets are built in:

| Dataset | `--public-dataset` | Auth Required | Description |
|---|---|---|---|
| LibriSpeech | `librispeech` | No | Read speech from audiobooks (test split) |
| VoxPopuli | `voxpopuli` | No | European Parliament recordings (Meta) |
| GigaSpeech | `gigaspeech` | Yes | Multi-domain corpus: audiobooks, podcasts, YouTube |
| AMI | `ami` | No | Meeting recordings with individual headset microphone audio |
| SPGISpeech | `spgispeech` | Yes | Financial earnings call recordings (Kensho) |

Clips longer than 30 seconds are automatically skipped to stay within typical ASR model context
limits.

---

## Start a vLLM Server

Launch vLLM with an audio-capable model such as Qwen2-Audio:

```bash
docker build -t vllm-audio - << 'EOF'
FROM vllm/vllm-openai:latest
RUN pip install 'vllm[audio]'
EOF

docker run --gpus all -p 8000:8000 vllm-audio \
  --model Qwen/Qwen2-Audio-7B-Instruct \
  --trust-remote-code
```

Verify the server is ready:

```bash
curl -s localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2-Audio-7B-Instruct","messages":[{"role":"user","content":"test"}],"max_tokens":1}'
```

---

## Profile with LibriSpeech

LibriSpeech is the standard read-speech benchmark and requires no authentication:

<!-- aiperf-run-vllm-audio-openai-endpoint-server-asr -->
```bash
aiperf profile \
    --model Qwen/Qwen2-Audio-7B-Instruct \
    --endpoint-type chat \
    --streaming \
    --url localhost:8000 \
    --public-dataset librispeech \
    --request-count 10 \
    --concurrency 4
```
<!-- /aiperf-run-vllm-audio-openai-endpoint-server-asr -->

**Sample Output:**

```

            NVIDIA AIPerf | LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━┓
┃                Metric ┃        avg ┃       min ┃        max ┃        p99 ┃        p90 ┃        p50 ┃       std ┃
┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━┩
│   Time to First Token │  31,838.96 │ 10,246.67 │  51,101.07 │  50,755.33 │  47,643.67 │  34,232.54 │ 11,974.17 │
│                  (ms) │            │           │            │            │            │            │           │
│  Time to Second Token │   1,612.02 │    348.35 │  11,892.33 │  10,868.74 │   1,656.45 │     468.33 │  3,427.09 │
│                  (ms) │            │           │            │            │            │            │           │
│  Time to First Output │  31,838.96 │ 10,246.67 │  51,101.07 │  50,755.33 │  47,643.67 │  34,232.54 │ 11,974.17 │
│            Token (ms) │            │           │            │            │            │            │           │
│  Request Latency (ms) │ 118,587.60 │ 25,920.68 │ 293,970.87 │ 281,381.83 │ 168,080.54 │ 110,066.27 │ 73,231.75 │
│   Inter Token Latency │   2,306.51 │    187.13 │   4,102.92 │   4,051.39 │   3,587.58 │   2,498.81 │  1,079.49 │
│                  (ms) │            │           │            │            │            │            │           │
│          Output Token │       0.95 │      0.24 │       5.34 │       4.95 │       1.42 │       0.40 │      1.48 │
│   Throughput Per User │            │           │            │            │            │            │           │
│     (tokens/sec/user) │            │           │            │            │            │            │           │
│      E2E Output Token │       0.37 │      0.19 │       0.90 │       0.87 │       0.59 │       0.29 │      0.20 │
│            Throughput │            │           │            │            │            │            │           │
│     (tokens/sec/user) │            │           │            │            │            │            │           │
│       Output Sequence │      37.10 │     10.00 │      78.00 │      75.21 │      50.10 │      39.50 │     18.10 │
│       Length (tokens) │            │           │            │            │            │            │           │
│ Input Sequence Length │       5.00 │      5.00 │       5.00 │       5.00 │       5.00 │       5.00 │      0.00 │
│              (tokens) │            │           │            │            │            │            │           │
│          Output Token │       1.24 │       N/A │        N/A │        N/A │        N/A │        N/A │       N/A │
│            Throughput │            │           │            │            │            │            │           │
│          (tokens/sec) │            │           │            │            │            │            │           │
│    Request Throughput │       0.03 │       N/A │        N/A │        N/A │        N/A │        N/A │       N/A │
│        (requests/sec) │            │           │            │            │            │            │           │
│         Request Count │      10.00 │       N/A │        N/A │        N/A │        N/A │        N/A │       N/A │
│            (requests) │            │           │            │            │            │            │           │
└───────────────────────┴────────────┴───────────┴────────────┴────────────┴────────────┴────────────┴───────────┘

```

> High TTFT variance is expected for ASR workloads — audio encoding time scales with clip duration.
> Clips vary in length (up to 30 seconds) so TTFT will vary across requests.

---

## Profile with VoxPopuli

VoxPopuli contains European Parliament recordings and requires no authentication:

```bash
aiperf profile \
    --model Qwen/Qwen2-Audio-7B-Instruct \
    --endpoint-type chat \
    --streaming \
    --url localhost:8000 \
    --public-dataset voxpopuli \
    --request-count 10 \
    --concurrency 4
```

---

## Profile with AMI

AMI contains meeting recordings with individual headset microphone audio and requires no authentication:

```bash
aiperf profile \
    --model Qwen/Qwen2-Audio-7B-Instruct \
    --endpoint-type chat \
    --streaming \
    --url localhost:8000 \
    --public-dataset ami \
    --request-count 10 \
    --concurrency 4
```

---

## Profile with GigaSpeech

GigaSpeech is a multi-domain corpus covering audiobooks, podcasts, and YouTube. It requires a
HuggingFace account and acceptance of the [dataset terms](https://huggingface.co/datasets/speechcolab/gigaspeech):

```bash
uv run hf auth login

aiperf profile \
    --model Qwen/Qwen2-Audio-7B-Instruct \
    --endpoint-type chat \
    --streaming \
    --url localhost:8000 \
    --public-dataset gigaspeech \
    --request-count 10 \
    --concurrency 4
```

---

## Profile with SPGISpeech

SPGISpeech contains financial earnings call recordings. It requires a HuggingFace account and
acceptance of the [dataset terms](https://huggingface.co/datasets/kensho/spgispeech):

```bash
uv run hf auth login

aiperf profile \
    --model Qwen/Qwen2-Audio-7B-Instruct \
    --endpoint-type chat \
    --streaming \
    --url localhost:8000 \
    --public-dataset spgispeech \
    --request-count 10 \
    --concurrency 4
```
