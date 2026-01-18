<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Profile Hugging Face TGI Models with AIPerf

AIPerf can benchmark **Large Language Models (LLMs)** served through the
[Hugging Face Text Generation Inference (TGI)](https://huggingface.co/docs/text-generation-inference)
`generate` API.
TGI exposes two standard HTTP endpoints for text generation:

| Endpoint | Description | AIPerf Flag |
|-----------|--------------|--------------|
| `/generate` | Returns the full text completion in one response (non-streaming). | *(default)* |
| `/generate_stream` | Streams generated tokens as they are produced (SSE). | `--streaming` |


## Start a Hugging Face TGI Server

To launch a Hugging Face TGI server, use the official `ghcr.io` image:

```bash
docker run --gpus all --rm -it \
  -p 8080:80 \
  -e MODEL_ID=TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  ghcr.io/huggingface/text-generation-inference:latest
```

```bash
# Verify the server is running
curl -s http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"inputs":"Hello world"}' | jq
```

## Profile with AIPerf

You can benchmark TGI models in either non-streaming or streaming,
and with either synthetic inputs or a custom input file.

### Non-Streaming (`/generate`)

#### Profile with synthetic inputs

```bash
aiperf profile \
    -m TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --endpoint-type huggingface_generate \
    --url localhost:8080 \
    --request-count 10
```

#### Profile with custom input file

You can also provide your own text prompts using the
--input-file option.
The file should be in JSONL format and contain text entries.

```bash
cat > inputs.jsonl <<'EOF'
{"text": "Hello TinyLlama!"}
{"text": "Tell me a joke."}
EOF
```
Then run:

```bash
aiperf profile \
    -m TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --endpoint-type huggingface_generate \
    --url localhost:8080 \
    --input-file ./inputs.jsonl \
    --custom-dataset-type single_turn \
    --request-count 10
```

### Streaming (`/generate_stream`)

When the `--streaming` flag is enabled, AIPerf automatically sends requests to the `/generate_stream` endpoint of the TGI server.

#### Profile with synthetic inputs

```bash
aiperf profile \
    -m TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --endpoint-type huggingface_generate \
    --url localhost:8080 \
    --streaming \
    --request-count 10
```

#### Profile with custom input file

Create your own prompt file in JSONL format:

```bash
cat > inputs.jsonl <<'EOF'
{"text": "Explain quantum computing in simple terms."}
{"text": "Write a haiku about rain."}
{"text": "Summarize the causes of the French Revolution."}
EOF
```

Then run:

```bash
aiperf profile \
    -m TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --endpoint-type huggingface_generate \
    --url localhost:8080 \
    --input-file ./inputs.jsonl \
    --custom-dataset-type single_turn \
    --streaming \
    --request-count 10
```
