<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Performance Tuning Guide

This guide helps you optimize AIPerf for maximum throughput, lowest latency, and efficient resource utilization.

## Table of Contents

- [Quick Wins](#quick-wins)
- [Worker Optimization](#worker-optimization)
- [Connection Pooling](#connection-pooling)
- [Memory Management](#memory-management)
- [Network Optimization](#network-optimization)
- [CPU Optimization](#cpu-optimization)
- [Environment Variable Tuning](#environment-variable-tuning)
- [Bottleneck Identification](#bottleneck-identification)
- [Scaling Considerations](#scaling-considerations)

## Quick Wins

Start here for immediate performance improvements:

### 1. Increase Worker Count

```bash
# Auto-scaled (default: ~75% of CPU cores)
aiperf profile ...

# Manual override for higher throughput
aiperf profile --workers-max 32 ...
```

**Impact:** 2-5x throughput increase on multi-core systems

### 2. Increase Connection Limit

```bash
export AIPERF_HTTP_CONNECTION_LIMIT=5000
aiperf profile --concurrency 100 ...
```

**Impact:** Eliminates connection pool bottlenecks at high concurrency

### 3. Increase Buffer Sizes

```bash
export AIPERF_HTTP_SO_RCVBUF=10485760  # 10MB
export AIPERF_HTTP_SO_SNDBUF=10485760  # 10MB
aiperf profile ...
```

**Impact:** Better streaming performance, especially for large responses

### 4. Use Warmup

```bash
aiperf profile --warmup-request-count 50 --request-count 500
```

**Impact:** 10-20% improvement in average latency (eliminates cold-start overhead)

### 5. Tune Yield Intervals

```bash
export AIPERF_ZMQ_PULL_YIELD_INTERVAL=10
export AIPERF_ZMQ_SUB_YIELD_INTERVAL=10
aiperf profile ...
```

**Impact:** Prevents event loop starvation at very high message rates

## Worker Optimization

### Understanding Worker Scaling

AIPerf auto-scales workers using this formula:
```
max_workers = min(concurrency, int(CPU_count × 0.75) - 1, MAX_WORKERS_CAP)
```

Default `MAX_WORKERS_CAP = 32`

### Choosing Worker Count

| Scenario | Worker Count | Reasoning |
|----------|--------------|-----------|
| **Low concurrency (< 10)** | Match concurrency | More workers = wasted resources |
| **Moderate (10-50)** | 8-16 workers | Balance between throughput and overhead |
| **High (50-200)** | 16-32 workers | Maximize throughput |
| **Very high (> 200)** | 32-64 workers | Requires system tuning (see below) |

```bash
# Manual override
aiperf profile --workers-max 32 --concurrency 100
```

### Worker CPU Utilization

Adjust the CPU utilization factor:

```bash
# Default: 75% of cores
export AIPERF_WORKER_CPU_UTILIZATION_FACTOR=0.75

# Aggressive (use 90% of cores)
export AIPERF_WORKER_CPU_UTILIZATION_FACTOR=0.9

# Conservative (use 50% of cores, leave room for other processes)
export AIPERF_WORKER_CPU_UTILIZATION_FACTOR=0.5
```

### Worker vs Record Processor Balance

Record processors compute metrics from worker results:

```bash
# Default: 1 record processor per 4 workers
export AIPERF_RECORD_PROCESSOR_SCALE_FACTOR=4

# More processors for metric-heavy workloads
export AIPERF_RECORD_PROCESSOR_SCALE_FACTOR=2

# Fewer processors for simple workloads
export AIPERF_RECORD_PROCESSOR_SCALE_FACTOR=8
```

**Rule of thumb:** If workers are idle but records are backing up, decrease `SCALE_FACTOR`.

## Connection Pooling

### Connection Limit Tuning

Default: 2,500 connections

```bash
# For high concurrency (> 500)
export AIPERF_HTTP_CONNECTION_LIMIT=5000

# For very high concurrency (> 2000)
export AIPERF_HTTP_CONNECTION_LIMIT=10000
```

**System limits to increase:**
```bash
# Increase file descriptor limit
ulimit -n 65535

# Make permanent (add to /etc/security/limits.conf)
* soft nofile 65535
* hard nofile 65535
```

### Connection Reuse Strategy

Choose based on your workload:

```bash
# Pooled (default, best performance)
aiperf profile --connection-reuse-strategy pooled

# Never (new connection per request, for testing)
aiperf profile --connection-reuse-strategy never

# Sticky (for multi-turn with load balancing)
aiperf profile --connection-reuse-strategy sticky-user-sessions
```

### Keepalive Tuning

Adjust connection keepalive for long-running benchmarks:

```bash
# Default: 300 seconds (5 minutes)
export AIPERF_HTTP_KEEPALIVE_TIMEOUT=300

# Longer keepalive for sustained benchmarks
export AIPERF_HTTP_KEEPALIVE_TIMEOUT=600

# Shorter for connection churn testing
export AIPERF_HTTP_KEEPALIVE_TIMEOUT=60
```

### DNS Caching

Enable DNS caching to reduce lookup overhead:

```bash
# Default: 300 seconds
export AIPERF_HTTP_TTL_DNS_CACHE=300

# Longer cache for stable environments
export AIPERF_HTTP_TTL_DNS_CACHE=3600

# Disable for dynamic DNS
export AIPERF_HTTP_TTL_DNS_CACHE=0
```

## Memory Management

### Reducing Memory Usage

If AIPerf consumes too much memory:

```bash
# 1. Reduce worker count
aiperf profile --workers-max 8

# 2. Reduce export batch sizes
export AIPERF_RECORD_EXPORT_BATCH_SIZE=50
export AIPERF_RECORD_RAW_EXPORT_BATCH_SIZE=5

# 3. Disable raw exports
aiperf profile --export-level records  # instead of 'raw'

# 4. Reduce metric array capacity
export AIPERF_METRICS_ARRAY_INITIAL_CAPACITY=5000
```

### Memory-Mapped Datasets

For very large custom datasets, use memory mapping:

```bash
# Use shared memory-mapped storage
export AIPERF_DATASET_MMAP_BASE_PATH=/dev/shm

# Or use a fast disk
export AIPERF_DATASET_MMAP_BASE_PATH=/mnt/nvme
```

**When to use:** Dataset > 1GB or Kubernetes with shared volumes

### Prefill Concurrency

Prevent OOM during long-context benchmarks:

```bash
# Limit concurrent prefill operations
aiperf profile \
  --concurrency 100 \
  --prefill-concurrency 4 \
  --isl 8192  # Long context
```

**Impact:** Prevents memory exhaustion with minimal throughput loss

## Network Optimization

### TCP Socket Tuning

Optimize TCP socket buffers for high throughput:

```bash
# Receive buffer (10MB default)
export AIPERF_HTTP_SO_RCVBUF=20971520  # 20MB for large responses

# Send buffer (10MB default)
export AIPERF_HTTP_SO_SNDBUF=20971520  # 20MB for large requests

# Socket timeouts
export AIPERF_HTTP_SO_RCVTIMEO=60  # 60 seconds
export AIPERF_HTTP_SO_SNDTIMEO=60
```

### TCP Keepalive

Fine-tune TCP keepalive probes:

```bash
# Start keepalive after 60s idle
export AIPERF_HTTP_TCP_KEEPIDLE=60

# Probe interval: 30s
export AIPERF_HTTP_TCP_KEEPINTVL=30

# Probe count before giving up: 1
export AIPERF_HTTP_TCP_KEEPCNT=1

# User timeout (Linux): 30s
export AIPERF_HTTP_TCP_USER_TIMEOUT=30000
```

### ZMQ Socket Tuning

Optimize internal message bus:

```bash
# Send/receive timeouts (5 minutes default)
export AIPERF_ZMQ_SNDTIMEO=300000  # milliseconds
export AIPERF_ZMQ_RCVTIMEO=300000

# TCP keepalive for ZMQ
export AIPERF_ZMQ_TCP_KEEPALIVE_IDLE=60
export AIPERF_ZMQ_TCP_KEEPALIVE_INTVL=10
```

### Request Timeout

Adjust for slow servers or large outputs:

```bash
# Default: 6 hours
aiperf profile --request-timeout-seconds 21600

# Shorter for fast servers
aiperf profile --request-timeout-seconds 60

# Longer for slow inference
aiperf profile --request-timeout-seconds 43200  # 12 hours
```

## CPU Optimization

### Disable Event Loop Monitoring

For absolute maximum performance, disable health checks:

```bash
export AIPERF_SERVICE_EVENT_LOOP_HEALTH_ENABLED=False
```

**Warning:** You won't get warnings about blocked event loops. Only use if you've already tuned and validated your workload.

### Use uvloop (Default)

AIPerf uses uvloop by default for ~30% faster async I/O:

```bash
# Verify uvloop is enabled (default)
export AIPERF_SERVICE_DISABLE_UVLOOP=False

# Only disable if you have issues
export AIPERF_SERVICE_DISABLE_UVLOOP=True
```

### Garbage Collection

Workers and TimingManager have GC disabled by default for lower latency. If you see memory growth:

```bash
# Re-enable GC (not recommended, but available)
# This is a service_metadata setting, not directly configurable via env var
# File an issue if you need this functionality
```

### CPU Affinity (Advanced)

For dedicated benchmark machines, pin workers to specific cores:

```bash
# Example: Pin to specific CPUs
taskset -c 0-15 aiperf profile ...
```

## Environment Variable Tuning

### Complete Performance Profile

For maximum throughput:

```bash
# Workers
export AIPERF_WORKER_CPU_UTILIZATION_FACTOR=0.9
export AIPERF_WORKER_MAX_WORKERS_CAP=64

# HTTP
export AIPERF_HTTP_CONNECTION_LIMIT=10000
export AIPERF_HTTP_SO_RCVBUF=20971520
export AIPERF_HTTP_SO_SNDBUF=20971520
export AIPERF_HTTP_KEEPALIVE_TIMEOUT=600

# ZMQ
export AIPERF_ZMQ_PULL_YIELD_INTERVAL=10
export AIPERF_ZMQ_SUB_YIELD_INTERVAL=10
export AIPERF_ZMQ_PULL_MAX_CONCURRENCY=200000

# Record Processing
export AIPERF_RECORD_PROCESSOR_SCALE_FACTOR=2
export AIPERF_RECORD_EXPORT_BATCH_SIZE=200

# Event Loop
export AIPERF_SERVICE_EVENT_LOOP_HEALTH_ENABLED=False

aiperf profile --workers-max 64 --concurrency 500
```

### Conservative Profile

For shared machines or when stability is critical:

```bash
# Workers
export AIPERF_WORKER_CPU_UTILIZATION_FACTOR=0.5
export AIPERF_WORKER_MAX_WORKERS_CAP=16

# HTTP
export AIPERF_HTTP_CONNECTION_LIMIT=2000

# ZMQ
export AIPERF_ZMQ_PULL_YIELD_INTERVAL=5

# Record Processing
export AIPERF_RECORD_PROCESSOR_SCALE_FACTOR=8

aiperf profile --workers-max 8 --concurrency 50
```

### Low-Latency Profile

Optimize for minimal latency over throughput:

```bash
# Fewer workers for less context switching
export AIPERF_WORKER_MAX_WORKERS_CAP=8

# Aggressive yielding
export AIPERF_ZMQ_PULL_YIELD_INTERVAL=1
export AIPERF_ZMQ_SUB_YIELD_INTERVAL=1

# Fast health checks
export AIPERF_SERVICE_EVENT_LOOP_HEALTH_INTERVAL=0.1

aiperf profile --workers-max 4 --concurrency 4
```

## Bottleneck Identification

### Symptoms and Diagnosis

| Symptom | Likely Bottleneck | Solution |
|---------|-------------------|----------|
| Workers at 100% CPU, low server load | Too few workers | Increase `--workers-max` |
| Low worker CPU, high network wait | Network latency | Check network, increase buffer sizes |
| High memory usage | Too many workers or large datasets | Reduce workers, use `--export-level records` |
| Event loop warnings | High message rate | Tune yield intervals |
| Connection pool exhaustion | Too many concurrent requests | Increase `CONNECTION_LIMIT` |

### Profiling AIPerf

Enable yappi profiling to find CPU bottlenecks:

```bash
# Requires: pip install yappi snakeviz
export AIPERF_DEV_ENABLE_YAPPI=True
aiperf profile ...

# View results
snakeviz yappi_output.prof
```

### Enable Debug Logging

Identify bottlenecks with detailed logs:

```bash
# Debug specific services
export AIPERF_DEV_DEBUG_SERVICES=worker,timing_manager

# Or use verbose flag
aiperf profile --verbose ...

# Or trace level (very verbose)
aiperf profile --extra-verbose ...
```

### Monitor System Resources

```bash
# During benchmark, monitor:
htop            # CPU and memory per process
iotop           # Disk I/O
nethogs         # Network usage per process
ss -s           # Socket statistics
```

### Analyze HTTP Trace Metrics

Enable trace metrics to identify network bottlenecks:

```bash
aiperf profile --show-trace-timing ...
```

Look at:
- `http_req_blocked`: Connection pool waits
- `http_req_dns_lookup`: DNS resolution time
- `http_req_connecting`: TCP/TLS handshake time
- `http_req_waiting`: Server processing time (TTFB)

See [HTTP Trace Metrics Tutorial](tutorials/http-trace-metrics.md).

## Scaling Considerations

### Single Machine Limits

Typical limits on a single machine:

| Resource | Soft Limit | Hard Limit | How to Increase |
|----------|------------|------------|-----------------|
| **Concurrency** | ~500 | ~2,000 | Tune system limits |
| **Request Rate** | ~5,000 QPS | ~20,000 QPS | More workers, tune network |
| **Workers** | 32 | 64 | Increase `MAX_WORKERS_CAP` |
| **Memory** | 16GB | 64GB+ | Use memory mapping |

### System Tuning for High Scale

For > 1,000 concurrency or > 10,000 QPS:

```bash
# 1. Increase file descriptors
ulimit -n 65535
# Make permanent in /etc/security/limits.conf

# 2. Increase ephemeral port range
sudo sysctl -w net.ipv4.ip_local_port_range="1024 65535"

# 3. Enable port reuse
sudo sysctl -w net.ipv4.tcp_tw_reuse=1

# 4. Increase max connections
sudo sysctl -w net.core.somaxconn=4096

# 5. Increase network buffers
sudo sysctl -w net.core.rmem_max=134217728
sudo sysctl -w net.core.wmem_max=134217728

# Make permanent in /etc/sysctl.conf
```

### Multi-Machine Scaling

For distributed benchmarking:

```bash
# Option 1: Multiple AIPerf instances
# Machine 1:
aiperf profile --url http://server:8000 --workers-max 32

# Machine 2:
aiperf profile --url http://server:8000 --workers-max 32

# Option 2: Load balancer
aiperf profile --url http://loadbalancer:8000 --workers-max 64

# Option 3: Client-side multi-URL
aiperf profile \
  --url http://server1:8000 \
  --url http://server2:8000 \
  --url-strategy round_robin \
  --workers-max 64
```

### Kubernetes Deployment

For large-scale benchmarking in Kubernetes:

See [Deployment Guide](deployment.md) for:
- Multi-pod AIPerf deployments
- Shared dataset volumes
- Resource limits and requests
- Network policy tuning

## Performance Checklist

Before running production benchmarks:

- [ ] System limits increased (file descriptors, ports)
- [ ] Worker count appropriate for CPU cores
- [ ] Connection limit matches concurrency
- [ ] Buffer sizes increased for large responses
- [ ] Warmup phase configured
- [ ] Yield intervals tuned for message rate
- [ ] Export level appropriate (don't export raw if not needed)
- [ ] Health monitoring configured or disabled
- [ ] Baseline performance established
- [ ] System resources monitored during run

## Benchmarking AIPerf Itself

To measure AIPerf's overhead:

```bash
# 1. Benchmark against mock server
# (eliminates server variability)

# 2. Compare with simple curl loop
for i in {1..100}; do
  curl -s http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"test","messages":[{"role":"user","content":"hi"}]}'
done

# 3. Profile with yappi
export AIPERF_DEV_ENABLE_YAPPI=True

# 4. Monitor with system tools
perf record -g aiperf profile ...
perf report
```

## See Also

- **[Best Practices](best-practices.md)** - Guidelines for effective benchmarking
- **[Environment Variables](environment_variables.md)** - Complete environment reference
- **[Troubleshooting](troubleshooting.md)** - Common performance issues
- **[Architecture](architecture.md)** - Understanding the system design
- **[Deployment Guide](deployment.md)** - Production deployments
