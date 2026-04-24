---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Template Endpoint
---

# Template Endpoint

The template endpoint provides a flexible way to benchmark custom APIs that don't match standard OpenAI formats. You define request payloads using Jinja2 templates and optionally specify how to extract responses using [JMESPath](https://jmespath.org/) queries.

## When to Use

Use the template endpoint when:
- Your API has a custom request/response format
- Standard endpoints (chat, completions, embeddings, rankings) don't fit your use case

## Basic Example

Benchmark an API that accepts text in a custom format:

```bash
aiperf profile \
  --model your-model \
  --url http://localhost:8000/custom-endpoint \
  --endpoint-type template \
  --extra-inputs payload_template:'
  {
    "text": {{ text|tojson }}
  }' \
  --synthetic-input-tokens-mean 100 \
  --output-tokens-mean 50 \
  --concurrency 4 \
  --request-count 20
```

**Sample Output (Successful Run):**
```
INFO     Starting AIPerf System
INFO     Using template endpoint with custom payload
INFO     AIPerf System is PROFILING

Profiling: 20/20 |████████████████████████| 100% [00:28<00:00]

INFO     Benchmark completed successfully
INFO     Results saved to: artifacts/your-model-template-concurrency4/

            NVIDIA AIPerf | LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┓
┃                      Metric ┃    avg ┃    min ┃    max ┃    p99 ┃    p50 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━┩
│        Request Latency (ms) │ 456.78 │ 389.23 │ 567.45 │ 554.32 │ 452.34 │
│    Time to First Token (ms) │  89.34 │  67.45 │ 112.34 │ 109.23 │  87.56 │
│    Inter Token Latency (ms) │  11.23 │   9.45 │  14.56 │  14.12 │  11.01 │
│ Output Token Count (tokens) │  50.00 │  48.00 │  52.00 │  51.89 │  50.00 │
│  Request Throughput (req/s) │   8.78 │      - │      - │      - │      - │
└─────────────────────────────┴────────┴────────┴────────┴────────┴────────┘

JSON Export: artifacts/your-model-template-concurrency4/profile_export_aiperf.json
```

## Configuration

Configure the template endpoint using `--extra-inputs`:

### Required

- **`payload_template`**: Jinja2 template defining the request payload format
  - Named template: `nv-embedqa`
  - File path: `/path/to/template.json`
  - Inline string: `'{"text": {{ text|tojson }}}'`

### Optional

- **`response_field`**: [JMESPath](https://jmespath.org/) query to extract data from responses
  - Auto-detection is used if not provided
  - Example: `data[0].embedding`

Any other `--extra-inputs` fields are merged into every request payload:

```bash
--extra-inputs temperature:0.7 top_p:0.9
```

## Template Variables

### Content Variables
- **`text`**: First text content (or `None`)
- **`texts`**: List of all text contents
- **`image`**, **`audio`**, **`video`**: First media content (or `None`)
- **`images`**, **`audios`**, **`videos`**: Lists of all media contents

### Named Content Variables
- **`query`**: First query text
- **`queries`**: All query texts
- **`passage`**: First passage text
- **`passages`**: All passage texts
- **`texts_by_name`**: Dict mapping content names to text lists
- **`images_by_name`**, **`audios_by_name`**, **`videos_by_name`**: Dicts for media

### Request Metadata
- **`model`**: Model name
- **`max_tokens`**: Output token limit
- **`stream`**: Whether streaming is enabled
- **`role`**: Message role
- **`turn`**: Current turn object
- **`turns`**: List of all turns
- **`request_info`**: Full request context

## Response Parsing

Auto-detection tries to extract in this order: embeddings, rankings, then text.

### Text Responses
- Fields: `text`, `content`, `response`, `output`, `result`
- OpenAI: `choices[0].text`, `choices[0].message.content`

### Embedding Responses
- OpenAI: `data[].embedding`
- Simple: `embeddings`, `embedding`

### Ranking Responses
- Lists: `rankings`, `results`

### Custom Extraction

Specify a [JMESPath](https://jmespath.org/) query to extract specific fields:

```bash
--extra-inputs response_field:'data[0].vector'
```

## Examples

### Custom Embedding API

```bash
aiperf profile \
  --model embedding-model \
  --url http://localhost:8000/embed \
  --endpoint-type template \
  --extra-inputs payload_template:'
    {
      "input": {{ texts|tojson }},
      "model": {{ model|tojson }}
    }' \
  --extra-inputs response_field:'embeddings' \
  --synthetic-input-tokens-mean 50 \
  --concurrency 8 \
  --request-count 100
```

### Named Template

Using the built-in `nv-embedqa` template:

```bash
aiperf profile \
  --model nv-embed-v2 \
  --url http://localhost:8000/embeddings \
  --endpoint-type template \
  --extra-inputs payload_template:nv-embedqa \
  --synthetic-input-tokens-mean 100 \
  --concurrency 4 \
  --request-count 50
```

**Note:** The `nv-embedqa` template expands to `{"text": {{ texts|tojson }}}`.

### Template from File

Create `chat_template.json`:

```jinja2
{
  "model": {{ model|tojson }},
  "prompt": {{ text|tojson }},
  "max_new_tokens": {{ max_tokens|tojson }},
  "stream": {{ stream|lower }}
}
```

Use it:

```bash
aiperf profile \
  --model custom-llm \
  --url http://localhost:8000/generate \
  --endpoint-type template \
  --extra-inputs payload_template:./chat_template.json \
  --extra-inputs response_field:'generated_text' \
  --streaming \
  --synthetic-input-tokens-mean 200 \
  --output-tokens-mean 100 \
  --concurrency 10
```

### Multi-Modal Request

```bash
aiperf profile \
  --model vision-model \
  --url http://localhost:8000/analyze \
  --endpoint-type template \
  --extra-inputs payload_template:'
    {
      "text": {{ text|tojson }},
      "image": {{ image|tojson }}
    }' \
  --input-file ./multimodal_dataset.jsonl \
  --concurrency 2
```

## YAML Config with Templates

Template endpoints work seamlessly with YAML configs, distributions, and sweeps. The Jinja2 template goes in `endpoint.template.body`.

### Basic YAML Config

```yaml
models:
  - custom-llm

endpoint:
  urls:
    - ${INFERENCE_URL:http://localhost:8000/generate}
  type: template
  streaming: true
  timeout: 120.0
  template:
    body: |
      {
        "model": {{ model|tojson }},
        "prompt": {{ text|tojson }},
        "max_new_tokens": {{ max_tokens|tojson }},
        "stream": {{ stream|lower }}
      }
    response_field: generated_text

datasets:
  main:
    type: synthetic
    entries: 500
    prompts:
      isl: {type: lognormal, mean: 512, sigma: 0.5}
      osl: {type: clamped, distribution: {type: normal, mean: 256, stddev: 80}, min: 16, max: 1024}

phases:
  profiling:
    type: poisson
    rate: 20.0
    duration: 120
    concurrency: 32
    grace_period: 30
```

### Custom Embedding API with Batch Distributions

```yaml
models:
  - embedding-model

endpoint:
  urls:
    - http://localhost:8000/embed
  type: template
  template:
    body: |
      {
        "input": {{ texts|tojson }},
        "model": {{ model|tojson }},
        "encoding_format": "float"
      }
    response_field: "data[0].embedding"

datasets:
  main:
    type: synthetic
    entries: 1000
    prompts:
      isl:
        type: mixture
        components:
          - distribution: {type: normal, mean: 64, stddev: 10}
            weight: 60
          - distribution: {type: normal, mean: 512, stddev: 50}
            weight: 40
      batch_size: 4

phases:
  profiling:
    type: concurrency
    concurrency: 16
    requests: 1000
    grace_period: 30
```

### Ranking API with Passage Distributions

```yaml
models:
  - rerank-model

endpoint:
  urls:
    - http://localhost:8000/rerank
  type: template
  template:
    body: |
      {
        "query": {{ query|tojson }},
        "passages": {{ passages|tojson }},
        "model": {{ model|tojson }}
      }
    response_field: "results"

datasets:
  main:
    type: synthetic
    entries: 500
    prompts:
      isl: {type: normal, mean: 32, stddev: 8}
    rankings:
      passages: {type: normal, mean: 10, stddev: 3}
      passage_tokens: {type: lognormal, mean: 128, sigma: 0.4}
      query_tokens: {type: normal, mean: 32, stddev: 8}

phases:
  profiling:
    type: poisson
    rate: 50.0
    duration: 120
    concurrency: 32
    grace_period: 30
```

### Sweep Across Template APIs

Use scenario sweeps to benchmark multiple custom API formats:

```yaml
endpoint:
  urls:
    - ${INFERENCE_URL:http://localhost:8000/v1}
  type: template
  template:
    body: |
      {
        "text": {{ text|tojson }},
        "model": {{ model|tojson }}
      }

datasets:
  main:
    type: synthetic
    entries: 500
    prompts:
      isl: {type: normal, mean: 256, stddev: 50}
      osl: 128

phases:
  profiling:
    type: poisson
    rate: 20.0
    duration: 60
    concurrency: 32

sweep:
  type: scenarios
  runs:
    - name: generate_endpoint
      endpoint:
        urls: ["http://localhost:8000/generate"]
        template:
          body: '{"prompt": {{ text|tojson }}, "max_tokens": {{ max_tokens|tojson }}}'
          response_field: generated_text

    - name: chat_endpoint
      endpoint:
        urls: ["http://localhost:8000/chat"]
        template:
          body: '{"messages": [{"role": "user", "content": {{ text|tojson }}}], "model": {{ model|tojson }}}'
          response_field: "choices[0].message.content"

    - name: completions_endpoint
      endpoint:
        urls: ["http://localhost:8000/completions"]
        template:
          body: '{"prompt": {{ text|tojson }}, "max_tokens": {{ max_tokens|tojson }}, "stream": {{ stream|lower }}}'

multi_run:
  num_runs: 3
  cooldown_seconds: 10.0
  confidence_level: 0.95

artifacts:
  dir: ./artifacts/template_sweep
  summary: [json]
```

<Note>
Jinja2 templates are only supported in the `endpoint.template.body` field for request payload formatting. YAML config values use environment variables (`${VAR:default}`) for dynamic substitution, not Jinja2 syntax.
</Note>

## Tips

- **Always use `|tojson`** for string/list values to properly escape JSON
- **Use `-v` or `-vv`** to see debug logs with formatted payloads
- **Check `artifacts/<run-name>/inputs.json`** to see all formatted request payloads
- **Let auto-detection work first** before specifying `response_field`

## Troubleshooting

**Template didn't render valid JSON**
- Use `|tojson` filter for string or nullable values

**Response not parsed correctly**
- Use `-vv` to see raw responses in logs
- Specify `response_field` with a [JMESPath](https://jmespath.org/) query

**Variables not available**
- Verify your input dataset includes the required fields
- Use `request_info` and `turn` objects for nested data
