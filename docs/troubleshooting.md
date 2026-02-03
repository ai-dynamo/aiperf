<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Troubleshooting Guide

This guide helps you diagnose and fix common issues with AIPerf.

## Table of Contents

- [Connection Issues](#connection-issues)
- [Configuration Errors](#configuration-errors)
- [Performance Problems](#performance-problems)
- [Worker Issues](#worker-issues)
- [Memory Problems](#memory-problems)
- [Plugin Errors](#plugin-errors)
- [Output and Export Issues](#output-and-export-issues)
- [GPU Telemetry Issues](#gpu-telemetry-issues)
- [Debugging Tips](#debugging-tips)
- [Reporting Bugs](#reporting-bugs)

## Connection Issues

### Cannot Connect to Server

**Symptoms:**
```
ERROR: Failed to connect to http://localhost:8000
ConnectionRefusedError: [Errno 111] Connection refused
```

**Causes & Solutions:**

1. **Server not running**
   ```bash
   # Check if server is listening
   netstat -tuln | grep 8000

   # Or try curl
   curl http://localhost:8000/health
   ```

2. **Wrong URL or port**
   ```bash
   # Verify server URL and port
   aiperf profile --url http://localhost:CORRECT_PORT ...
   ```

3. **Docker networking**
   ```bash
   # If AIPerf in container, use host.docker.internal
   --url http://host.docker.internal:8000

   # Or use --network=host
   docker run --network=host ...
   ```

4. **Firewall blocking connection**
   ```bash
   # Temporarily disable firewall to test
   sudo ufw disable  # Ubuntu
   sudo systemctl stop firewalld  # RHEL/CentOS
   ```

### Timeout Errors

**Symptoms:**
```
ERROR: Request timeout after 60.0 seconds
asyncio.exceptions.TimeoutError
```

**Solutions:**

```bash
# Increase request timeout
aiperf profile --request-timeout-seconds 300 ...

# Or via environment variable
export AIPERF_HTTP_SO_RCVTIMEO=300
```

### Connection Pool Exhausted

**Symptoms:**
```
WARNING: Connection pool limit reached
ERROR: Too many open connections
```

**Solutions:**

```bash
# Increase connection limit
export AIPERF_HTTP_CONNECTION_LIMIT=5000

# Or reduce concurrency
aiperf profile --concurrency 100 ...  # instead of 1000
```

### SSL Certificate Errors

**Symptoms:**
```
ERROR: SSL: CERTIFICATE_VERIFY_FAILED
```

**Solutions:**

```bash
# Disable SSL verification (NOT recommended for production)
export AIPERF_HTTP_SSL_VERIFY=False

# Or provide proper certificates
```

## Configuration Errors

### Invalid Model Name

**Symptoms:**
```
ERROR: Model 'model-name' not found on server
HTTP 404: Model not found
```

**Solutions:**

1. **Check exact model name on server**
   ```bash
   # For vLLM
   curl http://localhost:8000/v1/models

   # Match the name exactly
   aiperf profile --model "Qwen/Qwen3-0.6B" ...
   ```

2. **Try without organization prefix**
   ```bash
   # Some servers strip the org prefix
   --model "Qwen3-0.6B"  # instead of "Qwen/Qwen3-0.6B"
   ```

### Conflicting Options

**Symptoms:**
```
ERROR: Cannot specify both --concurrency and --user-centric-rate
ERROR: --request-count conflicts with --benchmark-duration
```

**Solutions:**

Read the error message carefully and choose one option:

```bash
# Use concurrency OR user-centric-rate, not both
aiperf profile --concurrency 10 ...

# Use request-count OR benchmark-duration, not both
aiperf profile --benchmark-duration 60 ...
```

### Invalid Dataset Configuration

**Symptoms:**
```
ERROR: --input-file requires --custom-dataset-type
ERROR: Cannot use both --public-dataset and --custom-dataset-type
```

**Solutions:**

```bash
# For custom datasets, specify type
aiperf profile \
  --input-file mydata.jsonl \
  --custom-dataset-type single_turn

# For public datasets, don't use --input-file
aiperf profile --public-dataset sharegpt
```

## Performance Problems

### Low Throughput

**Symptoms:**
- Request throughput much lower than expected
- High worker CPU usage but low server load

**Diagnosis:**

```bash
# Run with verbose logging
aiperf profile --log-level DEBUG ...

# Check worker count
ps aux | grep aiperf-worker | wc -l
```

**Solutions:**

1. **Increase worker count**
   ```bash
   # Auto-scales based on CPU
   aiperf profile --workers-max 32 ...
   ```

2. **Adjust concurrency**
   ```bash
   # Higher concurrency = more throughput
   aiperf profile --concurrency 50 ...
   ```

3. **Check network latency**
   ```bash
   # Measure baseline latency
   curl -w "@curl-format.txt" http://localhost:8000/v1/chat/completions
   ```

4. **Tune environment variables**
   ```bash
   export AIPERF_HTTP_CONNECTION_LIMIT=5000
   export AIPERF_HTTP_SO_RCVBUF=10485760
   export AIPERF_HTTP_SO_SNDBUF=10485760
   ```

### High Latency

**Symptoms:**
- TTFT or ITL much higher than expected
- Inconsistent latencies

**Solutions:**

1. **Use warmup phase**
   ```bash
   aiperf profile --warmup-request-count 50 ...
   ```

2. **Check server load**
   ```bash
   # Monitor server metrics
   aiperf profile --server-metrics ...
   ```

3. **Reduce concurrency**
   ```bash
   # Lower concurrency for lower latency
   aiperf profile --concurrency 4 ...
   ```

4. **Check for network issues**
   ```bash
   # Use HTTP trace metrics
   aiperf profile --show-trace-timing ...
   ```

### Event Loop Blocked

**Symptoms:**
```
WARNING: Event loop blocked for 100ms
WARNING: Event loop latency: 250ms
```

**Solutions:**

```bash
# Reduce worker count
aiperf profile --workers-max 8 ...

# Reduce concurrency
aiperf profile --concurrency 10 ...

# Adjust yield intervals
export AIPERF_ZMQ_PULL_YIELD_INTERVAL=5
export AIPERF_ZMQ_SUB_YIELD_INTERVAL=5
```

## Worker Issues

### Workers Not Starting

**Symptoms:**
```
ERROR: Failed to start workers
WARNING: Worker startup timeout
```

**Solutions:**

1. **Check system resources**
   ```bash
   # Check CPU and memory
   top
   free -h

   # Reduce worker count if needed
   aiperf profile --workers-max 4 ...
   ```

2. **Check ZMQ connectivity**
   ```bash
   # Use DEBUG logging
   aiperf profile --log-level DEBUG ...

   # Check for port conflicts
   netstat -tuln | grep LISTEN
   ```

3. **Try IPC instead of TCP**
   ```bash
   # For single-machine deployments
   export AIPERF_ZMQ_COMMUNICATION_BACKEND=ipc
   ```

### Worker Crashes

**Symptoms:**
```
ERROR: Worker PID 12345 died unexpectedly
WARNING: Worker Manager detected crashed worker
```

**Solutions:**

1. **Check worker logs**
   ```bash
   # Enable debug logging for workers
   export AIPERF_DEV_DEBUG_SERVICES=worker
   ```

2. **Reduce worker memory usage**
   ```bash
   # Reduce batch sizes
   export AIPERF_RECORD_EXPORT_BATCH_SIZE=50
   ```

3. **Check for OOM killer**
   ```bash
   # Check system logs
   dmesg | grep -i "out of memory"
   journalctl | grep -i oom
   ```

## Memory Problems

### Out of Memory (OOM)

**Symptoms:**
```
ERROR: CUDA out of memory
MemoryError: Cannot allocate memory
Killed (OOM killer)
```

**Solutions:**

1. **Reduce concurrency**
   ```bash
   aiperf profile --concurrency 4 ...
   ```

2. **Use prefill concurrency limit**
   ```bash
   aiperf profile --prefill-concurrency 2 ...
   ```

3. **Reduce sequence lengths**
   ```bash
   aiperf profile \
     --prompt-input-tokens-mean 256 \
     --prompt-output-tokens-mean 128
   ```

4. **Reduce worker count**
   ```bash
   aiperf profile --workers-max 4 ...
   ```

5. **For server OOM**, adjust server configuration:
   ```bash
   # vLLM example
   --max-model-len 2048 \
   --max-num-seqs 256
   ```

### Memory Leak

**Symptoms:**
- Memory usage continuously increases
- Performance degrades over time

**Solutions:**

```bash
# Enable GC for services (normally disabled for performance)
# This is a last resort - report as a bug

# Monitor memory with tools
python -m memory_profiler aiperf profile ...
```

## Plugin Errors

### Plugin Not Found

**Symptoms:**
```
ERROR: Type 'my_plugin' not found for category 'endpoint'
TypeNotFoundError
```

**Solutions:**

1. **Verify plugin is registered**
   ```bash
   aiperf plugins endpoint
   aiperf plugins endpoint my_plugin
   ```

2. **Check installation**
   ```bash
   # Reinstall package
   pip install -e .

   # Verify entry point
   pip show -v your-plugin-package
   ```

3. **Validate plugins**
   ```bash
   aiperf plugins --validate
   ```

### Plugin Import Error

**Symptoms:**
```
ERROR: Failed to import module for endpoint:my_plugin
ImportError: No module named 'my_module'
```

**Solutions:**

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Check class path**
   ```yaml
   # In plugins.yaml
   class: my_package.endpoints.my_endpoint:MyEndpointClass
   ```

3. **Test import manually**
   ```bash
   python -c "from my_package.endpoints.my_endpoint import MyEndpointClass"
   ```

## Output and Export Issues

### Missing Output Files

**Symptoms:**
- Expected files not in artifacts directory
- Empty output files

**Solutions:**

1. **Check export level**
   ```bash
   # Generate all outputs
   aiperf profile --export-level raw ...
   ```

2. **Check disk space**
   ```bash
   df -h
   ```

3. **Check permissions**
   ```bash
   ls -la artifacts/
   chmod 755 artifacts/
   ```

### Corrupted JSON/JSONL

**Symptoms:**
```
ERROR: Invalid JSON in profile_export.jsonl
json.decoder.JSONDecodeError
```

**Solutions:**

1. **Ensure benchmark completed**
   - Let benchmark finish naturally
   - Don't kill with SIGKILL

2. **Validate JSON**
   ```bash
   # Check for incomplete writes
   tail artifacts/*/profile_export.jsonl

   # Validate JSON
   jq . artifacts/*/profile_export.jsonl
   ```

## GPU Telemetry Issues

### DCGM Not Reachable

**Symptoms:**
```
WARNING: GPU telemetry unavailable: Could not reach DCGM endpoint
```

**Solutions:**

1. **Check DCGM is running**
   ```bash
   curl http://localhost:9400/metrics
   ```

2. **Specify custom endpoint**
   ```bash
   aiperf profile --gpu-telemetry http://custom-host:9400/metrics
   ```

3. **Disable if not needed**
   ```bash
   aiperf profile --no-gpu-telemetry
   ```

### Missing GPU Metrics

**Symptoms:**
- GPU telemetry file empty or incomplete
- Some metrics missing

**Solutions:**

```bash
# Check DCGM configuration
dcgmi profile --list

# Increase collection interval
export AIPERF_GPU_COLLECTION_INTERVAL=1.0

# Check DCGM exporter logs
docker logs dcgm-exporter
```

## Debugging Tips

### Enable Verbose Logging

```bash
# Debug level
aiperf profile --verbose ...

# Trace level (very verbose)
aiperf profile --extra-verbose ...

# Debug specific services
export AIPERF_DEV_DEBUG_SERVICES=worker,timing_manager
```

### Enable Developer Mode

```bash
# Show internal metrics and experimental features
export AIPERF_DEV_MODE=True
export AIPERF_DEV_SHOW_INTERNAL_METRICS=True
export AIPERF_DEV_SHOW_EXPERIMENTAL_METRICS=True
```

### Profile AIPerf Itself

```bash
# Profile with yappi
export AIPERF_DEV_ENABLE_YAPPI=True
aiperf profile ...

# View results
snakeviz yappi_output.prof
```

### Check System Resources

```bash
# CPU and memory
top
htop

# Network connections
netstat -tuln
ss -tuln

# Disk I/O
iotop

# Open files
lsof | grep aiperf
```

### Validate Configuration

```bash
# Validate plugins
aiperf plugins --validate

# Check environment
env | grep AIPERF

# Test server connectivity
curl -v http://localhost:8000/v1/models
```

## Reporting Bugs

If you've tried the solutions above and still have issues, please report a bug:

### Information to Include

1. **AIPerf version**
   ```bash
   aiperf --version
   python --version
   ```

2. **Full command**
   ```bash
   # Include complete aiperf command
   aiperf profile --model ... --url ... [all flags]
   ```

3. **Error messages**
   ```bash
   # Run with verbose logging
   aiperf profile --log-level DEBUG ... 2>&1 | tee debug.log
   ```

4. **Environment**
   ```bash
   # OS and Python info
   uname -a
   python --version
   pip list | grep aiperf

   # Environment variables
   env | grep AIPERF
   ```

5. **Server information**
   - Server type (vLLM, TGI, Triton, etc.)
   - Server version
   - Model being benchmarked

### Where to Report

- **GitHub Issues**: [https://github.com/ai-dynamo/aiperf/issues](https://github.com/ai-dynamo/aiperf/issues)
- **Discussions**: [https://github.com/ai-dynamo/aiperf/discussions](https://github.com/ai-dynamo/aiperf/discussions)
- **Discord**: [https://discord.gg/D92uqZRjCZ](https://discord.gg/D92uqZRjCZ)

### Before Reporting

1. Search existing issues for similar problems
2. Try with the latest version
3. Simplify your command to minimal reproduction case
4. Test against the example in the [Tutorial](tutorial.md)

## See Also

- **[Getting Started](getting-started.md)** - Installation and first benchmark
- **[FAQ](faq.md)** - Frequently asked questions
- **[CLI Options](cli_options.md)** - Complete command reference
- **[Environment Variables](environment_variables.md)** - Configuration options
- **[Best Practices](best-practices.md)** - Guidelines for effective benchmarking
