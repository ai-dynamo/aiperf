---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Benchmark Control Hooks
---

# Benchmark Control Hooks

AIPerf can POST control-plane requests to your inference server around a
benchmark cell: reset the KV cache once before the run, and start/stop a
server profiler around each **profiling** phase (not warmup).

Control traffic uses an isolated HTTP client. It is **not** recorded as
benchmark requests and does **not** appear in metrics or profile exports.

## Quick start

```bash
aiperf profile \
  --model your-model \
  --url http://localhost:8000 \
  --endpoint-type chat \
  --streaming \
  --concurrency 8 \
  --request-count 100 \
  --reset-kv-cache \
  --server-profiler
```

With defaults this issues:

| Hook | When | Default path |
|------|------|--------------|
| KV-cache reset | Once per logical cell, before warmup/profiling | `POST {origin}/reset_prefix_cache` |
| Profiler start | Before each `CreditPhase.PROFILING` runner | `POST {origin}/start_profile` |
| Profiler stop | After that profiling runner finishes | `POST {origin}/stop_profile` |

`{origin}` is `scheme://host:port` for each configured `--url`.

## YAML: `false | true | object`

Under `endpoint`, both hooks accept a bool or an object:

```yaml
endpoint:
  url: http://localhost:8000
  type: chat
  model_names: [your-model]
  # Disabled (default when omitted)
  reset_kv_cache: false
  server_profiler: false
```

```yaml
endpoint:
  url: http://localhost:8000
  type: chat
  model_names: [your-model]
  # Enabled with defaults
  reset_kv_cache: true
  server_profiler: true
```

```yaml
endpoint:
  url: http://localhost:8000
  type: chat
  model_names: [your-model]
  reset_kv_cache:
    path: /v1/admin/reset_prefix_cache
    timeout_seconds: 30
  server_profiler:
    start_path: /v1/admin/start_profile
    stop_path: /v1/admin/stop_profile
    timeout_seconds: 15
```

## CLI flags

| Flag | Maps to |
|------|---------|
| `--reset-kv-cache` | Enable reset with defaults |
| `--reset-kv-cache-path` | `endpoint.reset_kv_cache.path` |
| `--reset-kv-cache-timeout-seconds` | `endpoint.reset_kv_cache.timeout_seconds` |
| `--server-profiler` | Enable profiler with defaults |
| `--server-profiler-start-path` | `endpoint.server_profiler.start_path` |
| `--server-profiler-stop-path` | `endpoint.server_profiler.stop_path` |
| `--server-profiler-timeout-seconds` | `endpoint.server_profiler.timeout_seconds` |

Setting any path/timeout override also enables the corresponding hook
(same as passing `true` / `--reset-kv-cache` / `--server-profiler`).

## Defaults and relative paths

| Setting | Default |
|---------|---------|
| Reset path | `/reset_prefix_cache` |
| Profiler start path | `/start_profile` |
| Profiler stop path | `/stop_profile` |
| Timeouts | `endpoint.timeout` when unset |

Paths must be **relative** (start with `/`) and must not contain `://`.
AIPerf joins each path to every endpoint URL origin. Absolute URLs in
`path` / `start_path` / `stop_path` are rejected at config validation.

Control hooks require HTTP transport. Non-HTTP transports raise a
validation error when either hook is enabled.

## Lifecycle and ownership

```mermaid
sequenceDiagram
    participant Runner as CLI / MultiRun
    participant TM as TimingManager
    participant Orch as PhaseOrchestrator
    participant Srv as Inference server

    Runner->>Srv: POST reset (once per cell)
    Note over Runner,Srv: Fatal on failure
    TM->>Orch: prepare hooks + auth headers
    Orch->>Orch: warmup phase (no profiler)
    Orch->>Srv: POST start_profile
    Note over Orch,Srv: Fatal on start failure
    Orch->>Orch: profiling PhaseRunner.run()
    Orch->>Srv: POST stop_profile
    Note over Orch,Srv: Warning only on stop failure
```

- **Reset** runs in the CLI single-run / multi-run path once per logical
  benchmark cell (each sweep cell, each multi-trial subprocess), before
  services start issuing load.
- **Profiler** is owned by `TimingManager` / `PhaseOrchestrator`. Workers
  never fire control hooks.
- Warmup phases never start or stop the profiler.
- Multi-URL endpoints: control POSTs go to each **unique** origin
  (`scheme://host:port`); duplicate path-qualified URLs on the same host
  are deduplicated. A partial profiler-start failure best-effort stops
  already-started origins, then re-raises. Profiler stop attempts every
  unique origin and aggregates failures.
- **Seamless non-final profiling:** when a profiling phase has
  `seamless=True` and is not the last phase, profiler **start** still runs
  before send begins, but profiler **stop** is deferred until the phase
  drain callback (after in-flight credits return), not when `run()` returns
  at send-complete.

## Failure policy

| Hook | Failure behavior |
|------|------------------|
| `reset_kv_cache` | Fatal — abort the cell |
| Profiler start | Fatal — profiling does not begin |
| Profiler stop | Warning only — run completes |

Auth headers match readiness probes (`Authorization: Bearer …` or
Anthropic `x-api-key`, plus any `--header` / YAML `endpoint.headers`).

## Mock server

The in-repo mock server exposes countable admin routes for local
verification:

- `POST /reset_prefix_cache` (vLLM-style)
- `POST /flush_cache` (SGLang-style alias; same counter)
- `POST /start_profile`
- `POST /stop_profile`

These routes are outside the inference auth path and do not contribute
to benchmark records.

For per-server path mappings (vLLM, SGLang, TensorRT-LLM), see
[Control Hooks by Server](control-hooks-by-server.md).
