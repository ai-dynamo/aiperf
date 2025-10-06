<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Sequence Length Distributions for Advanced Benchmarking

Sequence length distributions allow you to specify complex patterns of input
sequence length (ISL) and output sequence length (OSL) pairs with different
probabilities and optional variance. This enables benchmarking of multiple
use cases such as summarization and Q&A on one endpoint.

## Overview

The sequence distribution feature provides benchmarking of mixed workloads
with different ISL and OSL pairings.

## Basic Usage

### Example command

Add variance to make workloads more realistic:

```bash
aiperf benchmark \
    --model nvidia/llama-3.1-nemotron-70b-instruct \
    --dataset synthetic \
    --sequence-distribution "256|20,128|10:60;512|40,256|20:40" \
    --concurrency 10 \
    --num-requests 100
```

This creates:
- 60% of requests with ISL ~ Normal(256, 20), OSL ~ Normal(128, 10)
- 40% of requests with ISL ~ Normal(512, 40), OSL ~ Normal(256, 20)

Values are automatically clamped to be at least 1.

## Supported Formats

### 1. Semicolon Format (Recommended)

**Basic:**
```
"ISL1,OSL1:PROB1;ISL2,OSL2:PROB2;..."
```

**With standard deviations:**
```
"ISL1|STDDEV1,OSL1|STDDEV1:PROB1;ISL2|STDDEV2,OSL2|STDDEV2:PROB2"
```

### 2. Bracket Format

**Basic:**
```
"[(ISL1,OSL1):PROB1,(ISL2,OSL2):PROB2]"
```

**With standard deviations:**
```
"[(256|10,128|5):60,(512|20,256|15):40]"
```

### 3. JSON Format

**Basic:**
```json
{"pairs": [{"isl": 256, "osl": 128, "prob": 60}, {"isl": 512, "osl": 256, "prob": 40}]}
```

**With standard deviations:**
```json
{"pairs": [
  {"isl": 256, "isl_stddev": 10, "osl": 128, "osl_stddev": 5, "prob": 60},
  {"isl": 512, "isl_stddev": 20, "osl": 256, "osl_stddev": 15, "prob": 40}
]}
```

## Practical Examples

### Example 1: Chatbot Workload Simulation

```bash
# Simulate typical chatbot traffic:
# - 70% short queries (quick questions)
# - 20% medium queries (explanations)
# - 10% long queries (complex tasks)

aiperf benchmark \
    --model nvidia/llama-3.1-nemotron-70b-instruct \
    --dataset synthetic \
    --sequence-distribution "64|10,32|8:70;256|40,128|20:20;1024|100,512|50:10" \
    --request-rate 2.0 \
    --duration 300s
```

### Example 2: Code Generation Workload

```bash
# Simulate code generation patterns:
# - 40% simple functions (short context, medium output)
# - 35% refactoring tasks (medium context, medium output)
# - 25% complex implementations (long context, long output)

aiperf benchmark \
    --model nvidia/llama-3.1-nemotron-70b-instruct \
    --dataset synthetic \
    --sequence-distribution "128|20,256|40:40;512|80,384|60:35;2048|200,1024|100:25" \
    --request-rate 1.0 \
    --duration 600s
```

### Example 3: Mixed Production Workload

```bash
# Replicate observed production patterns with realistic variance
aiperf benchmark \
    --model nvidia/llama-3.1-nemotron-70b-instruct \
    --dataset synthetic \
    --sequence-distribution "[(128|25,64|15):45,(512|100,256|50):30,(1024|150,512|80):20,(2048|300,1024|150):5]" \
    --concurrency 20 \
    --num-requests 1000
```

## Advanced Usage

### Working with Configuration Files

Create a configuration file `workload.jsonl`:

```json
{
  "model": "nvidia/llama-3.1-nemotron-70b-instruct",
  "dataset": "synthetic",
  "sequence_distribution": "256|20,128|10:50;512|40,256|20:30;1024|80,512|40:20",
  "request_rate": 2.0,
  "duration": "300s"
}
```

Run with:
```bash
aiperf benchmark --config workload.jsonl
```

### Validating Distributions

Test your distribution before running full benchmarks:

```bash
# Quick validation run
aiperf benchmark \
    --model nvidia/llama-3.1-nemotron-70b-instruct \
    --dataset synthetic \
    --sequence-distribution "your-distribution-here" \
    --num-requests 10 \
    --verbose
```

### Performance Considerations

- **Standard deviations** add computational overhead during sampling
- **Large distributions** (many pairs) may impact startup time slightly
- **Extreme variance** (large stddev) may produce unexpected sequence lengths

## Integration with Other Features

### With Custom Datasets

```bash
# Use distributions with custom prompts
aiperf benchmark \
    --model nvidia/llama-3.1-nemotron-70b-instruct \
    --dataset-path my_prompts.jsonl \
    --sequence-distribution "256|20,128|10:60;512|40,256|20:40"
```

### With Rate Limiting

```bash
# Combine with sophisticated scheduling
aiperf benchmark \
    --model nvidia/llama-3.1-nemotron-70b-instruct \
    --dataset synthetic \
    --sequence-distribution "256|15,128|8:80;1024|50,512|25:20" \
    --request-rate 1.5 \
    --ramp-up-duration 60s \
    --duration 300s
```

### With Multiple Endpoints

```bash
# Test distribution across different model sizes
aiperf benchmark \
    --model nvidia/llama-3.1-nemotron-70b-instruct \
    --endpoint-urls http://server1:8000/v1 http://server2:8000/v1 \
    --sequence-distribution "128|10,64|5:70;512|30,256|15:30" \
    --load-balancing round-robin
```

## Monitoring and Analysis

### Understanding Results

The benchmark results will show:

- **Average ISL/OSL**: Weighted averages across the distribution
- **ISL/OSL Variance**: How much actual values varied from means
- **Probability Adherence**: Whether the sampling matched expected ratios

### Key Metrics to Monitor

- **Latency by Sequence Length**: How performance varies with request size
- **Throughput Stability**: Whether mixed workloads affect overall throughput
- **Memory Usage**: Impact of varying sequence lengths on GPU memory
- **Queue Depth**: How request size distribution affects batching

## Best Practices

### Designing Realistic Distributions

1. **Analyze Production Logs**: Base distributions on real traffic patterns
2. **Start Simple**: Begin with 2-3 pairs, add complexity gradually
3. **Test Variance**: Use moderate standard deviations initially (10-20% of mean)
4. **Validate Probabilities**: Ensure probabilities sum to 100% or 1.0

### Performance Testing Strategy

1. **Baseline First**: Test with uniform distributions for comparison
2. **Gradual Complexity**: Start with basic distributions, add stddev later
3. **Monitor Extremes**: Watch for outlier sequence lengths affecting performance
4. **Statistical Significance**: Use enough requests for reliable probability sampling

### Common Pitfalls

- **Probability Sum**: Forgetting probabilities must sum to 100% (or 1.0)
- **Excessive Variance**: Too-large stddev creating unrealistic sequence lengths
- **Single-Pair Testing**: Using distributions with only one pair (use fixed ISL/OSL instead)
- **Format Confusion**: Mixing probability formats (percentages vs fractions)

## Troubleshooting

### Validation Errors

**"Probabilities must sum to 100.0"**
```bash
# Wrong: 256,128:50;512,256:40  (sums to 90)
# Right: 256,128:55;512,256:45   (sums to 100)
```

**"Invalid format"**
```bash
# Wrong: "256,128|10:60"  (stddev on wrong value)
# Right: "256|10,128:60"   (stddev after ISL)
```

### Performance Issues

**Slow sampling with stddev:**
- Reduce standard deviations or number of pairs
- Consider simpler distributions for high-throughput testing

**Unexpected sequence lengths:**
- Check that stddev values are reasonable relative to means
- Monitor for negative values being clamped to 1

### Getting Help

For advanced use cases or integration questions:
- Check the [AIPerf documentation](../index.md)
- Review [CLI options](../cli_options.md) for all available parameters
- See [architecture guide](../architecture.md) for implementation details