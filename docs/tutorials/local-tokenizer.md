<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Using Local Tokenizers Without HuggingFace

AIPerf can be configured to use local tokenizers without requiring a connection to HuggingFace. This is particularly useful in environments where direct access to HuggingFace is blocked or restricted.

This guide shows you how to run AIPerf using locally stored tokenizer files instead of downloading them from HuggingFace.

---

## Prerequisites

Before you begin, ensure you have:

- AIPerf installed
- Tokenizer files available locally (e.g., `tokenizer.json`, `vocab.txt`, `config.json`)
- A directory containing your tokenizer files in HuggingFace-compatible format

---

## Prepare Your Local Tokenizer

### 1. Organize Tokenizer Files

Make sure your tokenizer files are stored in a local directory. A typical tokenizer directory structure looks like this:

```
/path/to/your/local/tokenizer/
├── tokenizer.json
├── tokenizer_config.json
├── vocab.txt (or vocab.json)
└── config.json
```

### 2. Verify File Format

Ensure your tokenizer files match the HuggingFace/tokenizers format. The files should be compatible with the `transformers` library's tokenizer loading mechanism.

---

## Run AIPerf with Local Tokenizer

Use the `--tokenizer` parameter to specify the path to your local tokenizer directory or file:

```bash
aiperf profile \
    --tokenizer /path/to/your/local/tokenizer \
    --model your-model-name \
    --endpoint-type chat \
    --url localhost:8000 \
    --request-count 10
```

### Example: Using a Local Llama Tokenizer

```bash
aiperf profile \
    --tokenizer /home/user/tokenizers/llama-2-7b \
    --model llama-2-7b \
    --endpoint-type chat \
    --url localhost:8000 \
    --streaming \
    --request-count 20 \
    --concurrency 4
```

**Sample Output (Successful Run):**
```
INFO     Starting AIPerf System
INFO     Loading local tokenizer from: /home/user/tokenizers/llama-2-7b
INFO     Tokenizer loaded successfully (offline mode)
INFO     AIPerf System is PROFILING

Profiling: 20/20 |████████████████████████| 100% [00:42<00:00]

INFO     Benchmark completed successfully
INFO     Results saved to: artifacts/llama-2-7b-chat-concurrency4/

            NVIDIA AIPerf | LLM Metrics
┃                     Metric ┃     avg ┃     min ┃     max ┃     p99 ┃     p50 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━┩
│       Request Latency (ms) │ 1456.78 │ 1234.56 │ 1789.34 │ 1768.90 │ 1445.67 │
│   Time to First Token (ms) │  289.34 │  245.67 │  356.78 │  348.90 │  286.12 │
│   Inter Token Latency (ms) │   15.67 │   12.89 │   19.34 │   18.90 │   15.45 │
│ Request Throughput (req/s) │    6.78 │       - │       - │       - │       - │
└────────────────────────────┴─────────┴─────────┴─────────┴─────────┴─────────┘

JSON Export: artifacts/llama-2-7b-chat-concurrency4/profile_export_aiperf.json
```

### Example: Using a Local Qwen Tokenizer

```bash
aiperf profile \
    --tokenizer /opt/tokenizers/qwen-0.6b \
    --model Qwen/Qwen3-0.6B \
    --endpoint-type chat \
    --url localhost:8000 \
    --public-dataset sharegpt \
    --request-count 50
```

---

## Using Custom Tokenizers

If you are using a custom tokenizer (one that is not a standard pretrained model from HuggingFace), you can still use it with AIPerf as long as it adheres to the rules below.

### Requirement: HuggingFace Format

**Crucial**: Your custom tokenizer MUST be saved in the HuggingFace `transformers` format. AIPerf relies on the `transformers` library to load tokenizers, so standard compatibility is required.



---

## Important Notes

### No HuggingFace Connection Required

- When you specify a local path with `--tokenizer`, AIPerf loads the tokenizer directly from your local files
- **No files will be downloaded** from HuggingFace when using a local tokenizer path
- No internet connection to HuggingFace servers is required

### Fully Air-Gapped Environments

For strictly air-gapped environments where you want to explicitly forbid any connection attempts, you can set the following environment variables:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

This ensures that the underlying `transformers` library operates in offline mode.

### File Format Compatibility

- Your local tokenizer directory structure and files must match the HuggingFace/tokenizers format
- The tokenizer files should include standard files like `tokenizer.json`, `vocab.txt`, or `vocab.json`
- AIPerf uses the same tokenizer loading mechanism as the `transformers` library

### No Extra Flags Needed

- You do not need to set any additional flags unless your tokenizer requires custom code execution
- The `--tokenizer` parameter accepts both directory paths and direct file paths

---

## Troubleshooting

### Tokenizer Not Found

If you encounter errors about missing tokenizer files:

1. Verify the path you provided is correct
2. Check that the directory contains the required tokenizer files
3. Ensure file permissions allow AIPerf to read the files

### Incompatible Tokenizer Format

If the tokenizer fails to load:

1. Verify your tokenizer files are in HuggingFace-compatible format
2. Check that all required files are present (`tokenizer.json`, `config.json`, etc.)

