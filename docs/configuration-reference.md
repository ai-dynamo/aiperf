<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Configuration Reference

Complete guide to configuring AIPerf through CLI options, environment variables, and configuration patterns.

## Table of Contents

- [Configuration Methods](#configuration-methods)
- [Configuration Precedence](#configuration-precedence)
- [Complete Configuration Examples](#complete-configuration-examples)
- [Configuration Validation](#configuration-validation)
- [Template Configurations](#template-configurations)
- [Best Practices](#best-practices)

## Configuration Methods

AIPerf can be configured through three methods:

### 1. Command-Line Options

The primary configuration method. See [CLI Options](cli_options.md) for complete reference.

```bash
aiperf profile \
  --model your-model \
  --url http://localhost:8000 \
  --endpoint-type chat \
  --streaming \
  --concurrency 50 \
  --request-count 1000
```

**Advantages:**
- Explicit and visible
- Easy to version control (shell scripts)
- Self-documenting

**Use when:**
- Running ad-hoc benchmarks
- Testing different configurations
- Scripting benchmarks

### 2. Environment Variables

System-level configuration. See [Environment Variables](environment_variables.md) for complete reference.

```bash
# Set environment
export AIPERF_HTTP_CONNECTION_LIMIT=5000
export AIPERF_WORKER_MAX_WORKERS_CAP=32
export AIPERF_HTTP_SO_RCVBUF=10485760

# Run benchmark
aiperf profile --model your-model --url http://localhost:8000 ...
```

**Advantages:**
- Separate infrastructure config from benchmark config
- Share settings across multiple runs
- Easy deployment configuration (Docker, Kubernetes)

**Use when:**
- Deploying to production
- Container environments
- Shared infrastructure settings

### 3. Configuration Files (Shell Scripts)

Combine CLI options and environment variables in reusable scripts:

```bash
#!/bin/bash
# config/production.sh

# Infrastructure settings
export AIPERF_HTTP_CONNECTION_LIMIT=5000
export AIPERF_WORKER_MAX_WORKERS_CAP=32

# Benchmark settings
MODEL="your-model"
URL="http://prod-server:8000"
CONCURRENCY=50
REQUEST_COUNT=1000

# Run benchmark
aiperf profile \
  --model "$MODEL" \
  --url "$URL" \
  --endpoint-type chat \
  --streaming \
  --concurrency "$CONCURRENCY" \
  --request-count "$REQUEST_COUNT" \
  --random-seed 42 \
  --warmup-request-count 100
```

**Usage:**
```bash
bash config/production.sh
```

## Configuration Precedence

When the same setting is specified in multiple ways, AIPerf uses this precedence (highest to lowest):

```
1. Command-Line Options (highest priority)
2. Environment Variables
3. Default Values (lowest priority)
```

### Example

```bash
# Environment variable
export AIPERF_WORKER_MAX_WORKERS_CAP=16

# Command-line option overrides environment
aiperf profile --workers-max 32 ...

# Result: Uses 32 workers (CLI takes precedence)
```

### Precedence Rules

| Setting | CLI Option | Environment Variable | Default | Winner |
|---------|-----------|---------------------|---------|--------|
| Worker count | `--workers-max 32` | `AIPERF_WORKER_MAX_WORKERS_CAP=16` | Auto | **32** (CLI) |
| Connection limit | Not set | `AIPERF_HTTP_CONNECTION_LIMIT=5000` | 2500 | **5000** (Env) |
| Buffer size | Not set | Not set | 10MB | **10MB** (Default) |

## Complete Configuration Examples

### Basic Benchmark

Minimal configuration for quick tests:

```bash
aiperf profile \
  --model Qwen/Qwen3-0.6B \
  --url http://localhost:8000 \
  --endpoint-type chat \
  --streaming \
  --concurrency 10 \
  --request-count 100
```

### Production Load Test

Comprehensive production configuration:

```bash
#!/bin/bash
# production-benchmark.sh

# Infrastructure tuning
export AIPERF_HTTP_CONNECTION_LIMIT=5000
export AIPERF_WORKER_MAX_WORKERS_CAP=32
export AIPERF_HTTP_SO_RCVBUF=20971520
export AIPERF_HTTP_SO_SNDBUF=20971520
export AIPERF_WORKER_CPU_UTILIZATION_FACTOR=0.9
export AIPERF_RECORD_PROCESSOR_SCALE_FACTOR=2

# Benchmark configuration
aiperf profile \
  --model production-model \
  --url http://prod-lb:8000 \
  --endpoint-type chat \
  --streaming \
  \
  `# Load Pattern` \
  --concurrency 100 \
  --concurrency-ramp-duration 60 \
  --request-rate 200 \
  --arrival-pattern poisson \
  \
  `# Duration` \
  --benchmark-duration 600 \
  --warmup-duration 60 \
  \
  `# Workload` \
  --seq-dist "256,128:30;512,256:50;1024,512:20" \
  --random-seed 42 \
  \
  `# SLOs` \
  --goodput "time_to_first_token:50 inter_token_latency:10" \
  \
  `# Output` \
  --export-level records \
  --output-artifact-dir /results/prod-$(date +%Y%m%d-%H%M%S) \
  \
  `# Monitoring` \
  --gpu-telemetry dashboard \
  --server-metrics \
  --slice-duration 60
```

### Stress Test

High-load stress testing:

```bash
#!/bin/bash
# stress-test.sh

export AIPERF_HTTP_CONNECTION_LIMIT=10000
export AIPERF_WORKER_MAX_WORKERS_CAP=64
export AIPERF_ZMQ_PULL_YIELD_INTERVAL=10

aiperf profile \
  --model stress-test-model \
  --url http://test-server:8000 \
  --endpoint-type chat \
  --streaming \
  \
  --concurrency 500 \
  --prefill-concurrency 8 \
  --request-count 10000 \
  \
  --workers-max 64 \
  \
  --isl 2048 \
  --osl 512 \
  \
  --warmup-request-count 100 \
  \
  --export-level summary \
  --gpu-telemetry dashboard
```

### A/B Testing

Comparing two configurations:

```bash
#!/bin/bash
# ab-test.sh

SEED=42
MODEL="test-model"
CONCURRENCY=50
COUNT=1000

# Baseline
aiperf profile \
  --model "$MODEL" \
  --url http://baseline-server:8000 \
  --endpoint-type chat \
  --streaming \
  --concurrency "$CONCURRENCY" \
  --request-count "$COUNT" \
  --random-seed "$SEED" \
  --warmup-request-count 100 \
  --profile-export-prefix baseline

# Optimized
aiperf profile \
  --model "$MODEL" \
  --url http://optimized-server:8000 \
  --endpoint-type chat \
  --streaming \
  --concurrency "$CONCURRENCY" \
  --request-count "$COUNT" \
  --random-seed "$SEED" \
  --warmup-request-count 100 \
  --profile-export-prefix optimized

# Compare
aiperf plot --paths artifacts/baseline artifacts/optimized
```

### Custom Dataset Benchmark

Using custom prompts:

```bash
#!/bin/bash
# custom-dataset-benchmark.sh

aiperf profile \
  --model your-model \
  --url http://localhost:8000 \
  --endpoint-type chat \
  --streaming \
  \
  `# Custom dataset` \
  --input-file data/production-prompts.jsonl \
  --custom-dataset-type single_turn \
  --dataset-sampling-strategy sequential \
  \
  `# Load pattern` \
  --concurrency 50 \
  --warmup-request-count 50 \
  \
  `# Output` \
  --export-level raw \
  --output-artifact-dir results/custom-$(date +%Y%m%d)
```

### Long-Duration Stability Test

Overnight stability testing:

```bash
#!/bin/bash
# stability-test.sh

export AIPERF_HTTP_KEEPALIVE_TIMEOUT=600
export AIPERF_ZMQ_RCVTIMEO=600000

aiperf profile \
  --model stability-test-model \
  --url http://localhost:8000 \
  --endpoint-type chat \
  --streaming \
  \
  `# 8-hour test` \
  --concurrency 20 \
  --benchmark-duration 28800 \
  --warmup-duration 300 \
  \
  `# Analysis` \
  --slice-duration 600 \
  --random-seed 42 \
  \
  `# Monitoring` \
  --gpu-telemetry dashboard \
  --server-metrics \
  \
  `# Output` \
  --export-level records \
  --output-artifact-dir /results/stability-$(date +%Y%m%d)
```

## Configuration Validation

### Checking Configuration

Before running benchmarks, validate your configuration:

```bash
# 1. Check help for options
aiperf profile --help | grep your-option

# 2. Validate plugins
aiperf plugins --validate

# 3. Check environment variables
env | grep AIPERF

# 4. Dry-run with minimal requests
aiperf profile [your-config] --request-count 5
```

### Common Configuration Errors

**Missing Required Options:**
```bash
# ❌ Error: Missing required options
aiperf profile --concurrency 10

# ✅ Correct: Specify model and URL
aiperf profile --model M --url U --concurrency 10
```

**Conflicting Options:**
```bash
# ❌ Error: Cannot use both
aiperf profile --request-count 100 --benchmark-duration 60

# ✅ Correct: Use one or the other
aiperf profile --request-count 100
# OR
aiperf profile --benchmark-duration 60
```

**Invalid Values:**
```bash
# ❌ Error: Concurrency must be >= 1
aiperf profile --concurrency 0

# ✅ Correct
aiperf profile --concurrency 1
```

### Configuration Checklist

Before production benchmarks:

- [ ] Model name matches server
- [ ] URL is correct and reachable
- [ ] Endpoint type matches API
- [ ] Concurrency/request-rate appropriate
- [ ] Warmup configured
- [ ] Random seed set (for reproducibility)
- [ ] Output directory writable
- [ ] Workers count appropriate for CPU
- [ ] Connection limit sufficient
- [ ] Timeout values appropriate

## Template Configurations

### Environment-Specific Templates

#### Development

```bash
# config/dev.env
AIPERF_HTTP_CONNECTION_LIMIT=1000
AIPERF_WORKER_MAX_WORKERS_CAP=8
AIPERF_LOG_LEVEL=DEBUG
```

```bash
# dev-benchmark.sh
source config/dev.env
aiperf profile \
  --model dev-model \
  --url http://localhost:8000 \
  --concurrency 10 \
  --request-count 100 \
  --verbose
```

#### Staging

```bash
# config/staging.env
AIPERF_HTTP_CONNECTION_LIMIT=5000
AIPERF_WORKER_MAX_WORKERS_CAP=16
AIPERF_LOG_LEVEL=INFO
```

```bash
# staging-benchmark.sh
source config/staging.env
aiperf profile \
  --model staging-model \
  --url http://staging-server:8000 \
  --concurrency 50 \
  --benchmark-duration 300 \
  --warmup-duration 30
```

#### Production

```bash
# config/production.env
AIPERF_HTTP_CONNECTION_LIMIT=10000
AIPERF_WORKER_MAX_WORKERS_CAP=32
AIPERF_HTTP_SO_RCVBUF=20971520
AIPERF_HTTP_SO_SNDBUF=20971520
AIPERF_LOG_LEVEL=INFO
```

```bash
# production-benchmark.sh
source config/production.env
aiperf profile \
  --model production-model \
  --url http://prod-lb:8000 \
  --concurrency 100 \
  --benchmark-duration 600 \
  --warmup-duration 60 \
  --random-seed 42 \
  --goodput "time_to_first_token:50 inter_token_latency:10" \
  --gpu-telemetry dashboard \
  --server-metrics
```

### Use-Case Templates

#### Interactive Chat

```bash
# Optimize for low latency
aiperf profile \
  --model chat-model \
  --url http://localhost:8000 \
  --endpoint-type chat \
  --streaming \
  --concurrency 20 \
  --isl 256 \
  --osl 128 \
  --session-turns-mean 5 \
  --session-turn-delay-mean 2000 \
  --goodput "time_to_first_token:50 inter_token_latency:15"
```

#### Batch Processing

```bash
# Optimize for throughput
export AIPERF_WORKER_MAX_WORKERS_CAP=32
aiperf profile \
  --model batch-model \
  --url http://localhost:8000 \
  --endpoint-type completions \
  --concurrency 100 \
  --request-rate 200 \
  --benchmark-duration 600
```

#### Long-Context

```bash
# Safe long-context benchmarking
aiperf profile \
  --model long-context-model \
  --url http://localhost:8000 \
  --endpoint-type chat \
  --streaming \
  --isl 8192 \
  --osl 1024 \
  --concurrency 20 \
  --prefill-concurrency 4 \
  --warmup-request-count 10
```

## Best Practices

### Configuration Management

1. **Use Version Control**
   ```bash
   git add configs/
   git commit -m "Add production benchmark config"
   ```

2. **Document Configurations**
   ```bash
   # Add comments explaining choices
   # Infrastructure settings
   export AIPERF_HTTP_CONNECTION_LIMIT=5000  # Handles 100 concurrent
   ```

3. **Test Before Production**
   ```bash
   # Always test with small request count first
   aiperf profile [production-config] --request-count 10
   ```

4. **Keep Secrets Separate**
   ```bash
   # Never commit API keys
   export API_KEY=$(cat ~/.secrets/api_key)
   aiperf profile --api-key "$API_KEY" ...
   ```

### Configuration Patterns

**Incremental Tuning:**
```bash
# Start with defaults
aiperf profile --model M --url U --concurrency 10

# Add warmup
aiperf profile --model M --url U --concurrency 10 --warmup-request-count 20

# Add monitoring
aiperf profile --model M --url U --concurrency 10 --warmup-request-count 20 --gpu-telemetry

# Add realistic workload
aiperf profile --model M --url U --concurrency 10 --warmup-request-count 20 --gpu-telemetry --seq-dist "..."
```

**Environment-Aware Scripts:**
```bash
#!/bin/bash
ENV=${ENV:-dev}  # Default to dev
source config/${ENV}.env
aiperf profile [options...]
```

**Usage:**
```bash
ENV=staging ./benchmark.sh
ENV=production ./benchmark.sh
```

## See Also

- **[CLI Options](cli_options.md)** - Complete command reference
- **[Environment Variables](environment_variables.md)** - All environment settings
- **[Best Practices](best-practices.md)** - Benchmarking guidelines
- **[Examples](examples.md)** - Real-world configurations
- **[Performance Tuning](performance-tuning.md)** - Optimization guide
