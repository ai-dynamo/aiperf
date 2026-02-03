<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Frequently Asked Questions (FAQ)

## General Questions

### What is AIPerf?

AIPerf is a comprehensive benchmarking tool for measuring the performance of generative AI models. It provides detailed metrics on throughput, latency, token statistics, and resource utilization for LLM inference servers.

### How is AIPerf different from GenAI-Perf?

AIPerf is the next-generation evolution of GenAI-Perf with:
- **Better Architecture**: Multi-process design with distributed workers
- **More Features**: User-centric timing, prefill concurrency, goodput metrics, HTTP trace metrics
- **Better Performance**: Scalable to higher request rates
- **Extensibility**: Plugin system for custom endpoints and strategies
- **Better UX**: Real-time dashboard, better progress reporting

See [Migrating from GenAI-Perf](migrating.md) for detailed comparison.

### When should I use AIPerf vs other tools?

Use **AIPerf** when you need:
- Comprehensive LLM-specific metrics (TTFT, ITL, token throughput)
- High request rates and sustained load testing
- Multi-turn conversation benchmarking
- Custom load patterns (Poisson, gamma distributions)
- SLO-based goodput measurement
- Detailed per-request and aggregate metrics

Use **other tools** when you need:
- **k6/Locust**: General HTTP load testing without LLM-specific metrics
- **llmperf**: Simple single-script benchmarks
- **vLLM's benchmark script**: Quick throughput tests for vLLM only

### Is AIPerf open source?

Yes, AIPerf is open source under the Apache 2.0 license.

## Installation & Setup

### What are the system requirements?

**Minimum:**
- Python 3.10+
- 2 CPU cores
- 4GB RAM
- Network access to inference server

**Recommended:**
- 4+ CPU cores for multi-worker operation
- 8GB+ RAM
- Linux or macOS (Windows via WSL2)

### Does AIPerf work on Windows?

AIPerf works on Windows via WSL2 (Windows Subsystem for Linux). Native Windows support is not currently available due to multi-processing limitations.

### Can I use AIPerf with Docker?

Yes! AIPerf provides official Docker images:
```bash
docker pull aidynamo/aiperf:latest
```

See [Deployment Guide](deployment.md) for details.

## Benchmark Configuration

### How many requests should I send?

It depends on your goal:

- **Quick test**: 50-100 requests
- **Stable metrics**: 200-500 requests
- **Statistical significance**: 1000+ requests
- **Sustained load**: Use `--benchmark-duration` instead (e.g., 300 seconds)

More requests provide more stable statistics but take longer to complete.

### What concurrency level should I use?

Choose based on your use case:

- **Single-user latency**: `--concurrency 1`
- **Light load**: `--concurrency 4-8`
- **Moderate load**: `--concurrency 10-50`
- **Heavy load**: `--concurrency 100-500`
- **Stress testing**: `--concurrency 1000+` with `--prefill-concurrency` limits

Start low and increase gradually while monitoring server behavior.

### Should I use concurrency or request-rate mode?

**Use concurrency when:**
- You want to maintain constant in-flight requests
- Simulating concurrent users
- Server has request queuing
- You want maximum sustained throughput

**Use request-rate when:**
- You want to control arrival timing
- Simulating realistic traffic patterns
- Testing server capacity at specific rates
- Measuring latency under controlled load

You can combine both with `--request-rate --concurrency` for hybrid control.

### How do I choose input/output sequence lengths?

**For realistic workloads:**
- Check your application's actual prompt/response distributions
- Use `--seq-dist` for mixed workloads

**For capacity testing:**
- **Short prompts** (< 256 tokens): Tests decode throughput
- **Long prompts** (> 1024 tokens): Tests prefill throughput
- **Mixed**: Use `--isl 512 --isl-stddev 128` for variability

See [Sequence Distributions](tutorials/sequence-distributions.md) for advanced configuration.

## Metrics & Interpretation

### What's the difference between TTFT and TTFO?

- **TTFT (Time to First Token)**: Time until ANY first token (including reasoning tokens)
- **TTFO (Time to First Output Token)**: Time until first non-reasoning output token

For non-reasoning models, TTFT and TTFO are equivalent. For reasoning-capable models (DeepSeek-R1, Qwen3, etc.), TTFO aligns with GenAI-Perf's TTFT.

### Why are my performance metrics not reproducible?

AIPerf provides **deterministic dataset generation** with `--random-seed`, but performance metrics vary due to:
- Network timing variability
- Async I/O scheduling
- Worker load balancing (ZMQ)
- Server-side non-determinism

Dataset content (prompts, images) is reproducible; timing metrics are not. See [Reproducibility](reproducibility.md).

### What is "goodput" and when should I use it?

Goodput measures the throughput of requests meeting your Service Level Objectives (SLOs):

```bash
aiperf profile --goodput "time_to_first_token:50 inter_token_latency:10"
```

Use goodput when:
- You have specific latency requirements
- Comparing systems for quality-adjusted throughput
- Capacity planning with SLA constraints

See [Goodput Tutorial](tutorials/goodput.md).

### Why is my token count different from the server's?

Token count discrepancies can occur due to:
- Different tokenizers (client vs server)
- Special tokens (BOS, EOS, etc.)
- Chat templates adding system tokens

Use `--use-server-token-count` to use server-reported counts, or check `usage_diff` metrics in exports.

## Performance & Scaling

### How many workers should I use?

AIPerf auto-scales workers based on CPU count: `min(concurrency, CPU_count * 0.75) - 1`

You can override with `--workers-max`:
- More workers = higher potential throughput
- Each worker consumes ~100-200MB RAM
- Diminishing returns beyond CPU count
- Monitor CPU usage and adjust

### AIPerf is using 100% CPU. Is this normal?

Yes! AIPerf is designed for high throughput and will use available CPU for:
- Multiple worker processes
- Async I/O operations
- Metric computation
- ZMQ message routing

If you want to limit CPU usage, reduce `--workers-max` or `--concurrency`.

### Can AIPerf scale beyond a single machine?

Currently, AIPerf is designed for single-machine operation. For distributed benchmarking:
- Run multiple AIPerf instances against different servers
- Use load balancer with `--url http://loadbalancer:8000`
- Use `--multi-url` for client-side load balancing

Kubernetes-native distributed mode is planned for a future release.

### How do I benchmark very high request rates (10K+ QPS)?

1. **Increase workers**: `--workers-max 64`
2. **Tune connection pooling**: `export AIPERF_HTTP_CONNECTION_LIMIT=10000`
3. **Use request-rate mode**: `--request-rate 10000`
4. **Tune buffer sizes**: See [Performance Tuning](performance-tuning.md)
5. **Monitor resources**: Ensure system has CPU/network capacity

## Features & Capabilities

### Does AIPerf support streaming?

Yes! Use `--streaming` to enable Server-Sent Events (SSE) streaming and measure TTFT, TTST, ITL, and per-token metrics.

### Can I benchmark non-OpenAI APIs?

Yes! AIPerf supports:
- **Built-in**: OpenAI, HuggingFace TGI, Cohere, NIM, Solido
- **Custom**: Use `--endpoint-type template` with Jinja2 templates
- **Plugin**: Create custom endpoint plugins

See [Template Endpoint Tutorial](tutorials/template-endpoint.md) and [Plugin System](plugins/plugin-system.md).

### Does AIPerf support multi-turn conversations?

Yes! Configure with:
```bash
aiperf profile \
  --num-sessions 100 \
  --session-turns-mean 5 \
  --session-turn-delay-mean 1000
```

Or use ShareGPT dataset:
```bash
aiperf profile --public-dataset sharegpt --num-sessions 50
```

See [Multi-Turn Tutorial](tutorials/multi-turn.md).

### Can I use my own custom dataset?

Yes! AIPerf supports several custom formats:

```bash
# Single-turn JSONL
aiperf profile \
  --input-file prompts.jsonl \
  --custom-dataset-type single_turn

# Multi-turn conversations
aiperf profile \
  --input-file conversations.jsonl \
  --custom-dataset-type multi_turn

# Trace replay with timestamps
aiperf profile \
  --input-file trace.jsonl \
  --custom-dataset-type mooncake_trace
```

See [Custom Prompt Benchmarking](tutorials/custom-prompt-benchmarking.md).

### Does AIPerf support embeddings and rankings?

Yes! AIPerf supports:
- OpenAI embeddings (`--endpoint-type embeddings`)
- NIM embeddings (`--endpoint-type nim_embeddings`)
- Cohere rankings (`--endpoint-type cohere_rankings`)
- HuggingFace TEI rankings (`--endpoint-type hf_tei_rankings`)
- NIM rankings (`--endpoint-type nim_rankings`)

See [Embeddings](tutorials/embeddings.md) and [Rankings](tutorials/rankings.md) tutorials.

### Can I collect GPU metrics?

Yes! AIPerf integrates with DCGM for GPU telemetry:

```bash
aiperf profile --gpu-telemetry dashboard ...
```

Requires DCGM Exporter running on the server. See [GPU Telemetry Tutorial](tutorials/gpu-telemetry.md).

### Does AIPerf support multimodal (vision) models?

Yes! AIPerf can generate synthetic images, audio, and video:

```bash
# Vision models
aiperf profile \
  --endpoint-type chat \
  --image-batch-size 1 \
  --image-width-mean 512 \
  --image-height-mean 512

# Audio models
aiperf profile \
  --endpoint-type chat \
  --audio-batch-size 1 \
  --audio-length-mean 5.0
```

See [Vision Tutorial](tutorials/vision.md).

## Output & Exports

### What output files does AIPerf generate?

**Summary files:**
- `profile_export_aiperf.csv` - Aggregate metrics in CSV
- `profile_export_aiperf.json` - Aggregate metrics in JSON with metadata

**Detailed files** (with `--export-level records` or `raw`):
- `profile_export.jsonl` - Per-request metrics
- `profile_export_raw.jsonl` - Raw request/response data (with `--export-level raw`)
- `profile_export_gpu_telemetry.jsonl` - GPU metrics time series
- `profile_export_server_metrics.jsonl` - Server metrics time series

### How do I parse AIPerf output programmatically?

Use the Pydantic models provided:

```python
from aiperf.exporters.models import ProfileExportRecord
import orjson

with open("profile_export.jsonl") as f:
    for line in f:
        record = ProfileExportRecord(**orjson.loads(line))
        print(f"Latency: {record.request_latency}ms")
```

See [Working with Profile Exports](tutorials/working-with-profile-exports.md).

### Can I export to databases?

AIPerf doesn't have direct database exports, but you can:
1. Parse JSONL output with Python
2. Load CSV into pandas/SQL
3. Create custom exporter plugins
4. Use [Parquet exports](server_metrics/server-metrics.md) for efficient querying

## Common Issues

### AIPerf hangs at startup with invalid configuration

This is a known issue. If AIPerf hangs indefinitely:
1. Press Ctrl+C to terminate
2. Check configuration for typos
3. Validate CLI options against `aiperf profile --help`
4. Run with `--log-level DEBUG` to see where it hangs

See [Known Issues in README](../README.md#known-issues).

### I see port exhaustion errors at high concurrency

When using >15,000 concurrent connections:
1. Increase system limits:
   ```bash
   # Increase max file descriptors
   ulimit -n 65535

   # Increase ephemeral port range
   sudo sysctl -w net.ipv4.ip_local_port_range="1024 65535"

   # Enable port reuse
   sudo sysctl -w net.ipv4.tcp_tw_reuse=1
   ```

2. Use connection pooling effectively (enabled by default)
3. Consider reducing concurrency or using multiple AIPerf instances

### Metrics don't match my server's logs

Possible causes:
- **Clock skew**: Ensure clocks are synchronized
- **Different definitions**: Server may count metrics differently
- **Tokenizer mismatch**: Use `--use-server-token-count`
- **Measurement window**: Server may measure different time spans

## Getting Help

### Where can I find more examples?

- [Tutorial](tutorial.md) - Complete examples
- [Examples Gallery](examples.md) - Real-world scenarios
- [Tutorials Directory](tutorials/) - 31 feature-specific guides

### How do I contribute to AIPerf?

See [CONTRIBUTING.md](../CONTRIBUTING.md) for:
- Development setup
- Code style guidelines
- PR process
- Testing requirements

### Where do I report bugs?

- **GitHub Issues**: [https://github.com/ai-dynamo/aiperf/issues](https://github.com/ai-dynamo/aiperf/issues)
- **Discord**: [https://discord.gg/D92uqZRjCZ](https://discord.gg/D92uqZRjCZ)

See [Troubleshooting Guide](troubleshooting.md) first.

## See Also

- **[Getting Started](getting-started.md)** - Installation and first benchmark
- **[Troubleshooting](troubleshooting.md)** - Common issues and solutions
- **[Best Practices](best-practices.md)** - Guidelines for effective benchmarking
- **[CLI Options](cli_options.md)** - Complete command reference
- **[Glossary](glossary.md)** - Definitions of key terms
