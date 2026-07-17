<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Dynamo online replay parity (`replay_mode=online`)

AIPerf can replay a trace against Dynamo's passive perf-model engine under the
**real wall clock, in-process** — the equivalent of Dynamo's
`--replay-mode online`, with no sockets, HTTP, frontend, or mocker trace driver.
This is selected through the `dynosim` backend's `replay_mode` field:

```yaml
backend:
  type: dynosim
  config:
    replay_mode: online   # offline (default) = deterministic virtual clock
    engine: { block_size: 16 }
```

The correctness bar is an **apples-to-apples gate**: replaying the *same* trace
through AIPerf's online path and through Dynamo's own native online replay must
agree — request/token counts exact, every latency mean within **3%**, and
AIPerf throughput **≥** native. Both sides run under the real clock and measure
real wall-clock latency (`Instant::elapsed`), so the comparison is valid.

## The two code paths (end-to-end, both subprocesses)

Both processes replay the **same mooncake hash-block trace file**, feed the
**same passive engine** (linked as a library — never a subprocess/HTTP/Python
of its own), and produce byte-identical tokens because both derive them from the
same `hash_ids` via Dynamo's own `TurnTrace::synthesize_tokens`.

```
  SAME INPUT: one mooncake trace file (timestamp/input_length/output_length/hash_ids)
  SAME ENGINE ARGS: {"block_size":16}          SAME real wall clock, real-latency measurement

┌──────────────────────────────────────────┐   ┌──────────────────────────────────────────┐
│ (1) AIPERF PRODUCT PATH  [subprocess]      │   │ (2) NATIVE DYNAMO  [subprocess]            │
│     the `aiperf --execute` engine          │   │     the real native CLI                    │
└──────────────────────────────────────────┘   └──────────────────────────────────────────┘

  protocol-v2 JSON on stdin                        python -m dynamo.replay <trace>
        │                                                --replay-mode online
        ▼                                                --replay-concurrency N
  aiperf --execute  ENGINE                               --extra-engine-args '{"block_size":16}'
        │  backend=dynosim                        --report-json <out>
        │  replay_mode=online                                  │
        ▼                                                       ▼
  PreparedDynosimScheduledOperation                dynamo.replay.main
        ▼                                                       ▼
  DynosimExecutor  (replay_mode branch)            run_live_runtime
        ▼                                                       ▼
  run_scheduled_backend_online_deferred_with_delivery    ┌──────────────────────────────────┐
        ▼                                                 │ DYNAMO's OWN FLOW                 │
  ┌──────────────────────────────────┐                   │  • LiveRuntime                   │
  │ AIPERF's OWN FLOW                 │                   │  • its arrival/concurrency demux │
  │  • ScheduledRuntime / SlotPool    │                   │  • mpsc DirectRequest channels   │
  │  • materialize trace hash_ids     │                   │  • its own collector             │
  │    → DynamoTraceHashEncoder       │                   │  • tokio real-clock reactor      │
  │      (TurnTrace::synthesize_tokens)│  ── same conv ──▶ │    (sleep_until deadlines)        │
  │  • DynosimSink.dispatch     │                   └──────────────────────────────────┘
  │  • RequestObserver / collector    │                          │
  │  • drive_real_with_source ────────┼──┐               ┌───────┘  steps engine, sleeps real time
  │    (aiperf_runtime::graph real-clock │  │             │
  │     steps at event's sim-time,    │  │               │
  │     sleeps to deadline real time) │  ▼               ▼
  └──────────────────────────────────┘  ┌───────────────────────────────────────┐
        │                                │  SAME passive Dynamo perf-model engine │
        │                                │  SteppableReplay (SteppableAgg)        │
        │                                │  linked LIBRARY, computes token timings│
        │                                └───────────────────────────────────────┘
        ▼                                                    ▼
  native-v2.json                                       dynamo report.json
  (AIPerf observer-measured, real clock)               (Dynamo-measured, real clock)
        │                                                    │
        └──────────────────────────┬─────────────────────────┘
                                   ▼
                     GATE (report vs report):
                     • input/output tokens  EXACT
                     • ttft / e2e / itl mean  within 3%
                     • AIPerf throughput  >= native
```

### What each side owns

| | AIPerf online (path 1) | Native Dynamo (path 2) |
|---|---|---|
| Process | `aiperf --execute` (the `aiperf` binary) | `python -m dynamo.replay` |
| Trace driving | **AIPerf's own** `ScheduledRuntime`/`SlotPool`/`DynosimSink` | **Dynamo's own** `LiveRuntime`/demux/channels |
| Real-clock pump | `aiperf_runtime::graph::drive_real_with_source` | Dynamo's tokio reactor + `sleep_until` |
| Tokens | `hash_ids → TurnTrace::synthesize_tokens` | `hash_ids → TurnTrace::synthesize_tokens` |
| Engine | passive `SteppableReplay` (library) | **same** passive `SteppableReplay` (library) |
| Latency measurement | AIPerf observer, real clock | `Instant::elapsed`, real clock |

AIPerf drives its side **entirely**; it never calls Dynamo's `simulate_*` /
`LiveRuntime` / `WorkloadDriver` / trace driver. The native side is Dynamo's
genuine driver. Neither uses sockets, HTTP, or a subprocess *for the engine* —
the engine is a linked library on both sides.

## Measured parity (16 requests, ISL 256, OSL 8, block_size 16)

| Metric | AIPerf online | `python -m dynamo.replay` online | Δ |
|---|---|---|---|
| input tokens | 4096 | 4096 | exact |
| output tokens | 128 | 128 | exact |
| mean TTFT | 92.745 ms | 93.424 ms | 0.7% |
| mean e2e latency | 139.152 ms | 139.972 ms | 0.6% |
| mean ITL | 6.630 ms | 6.650 ms | 0.3% |
| request throughput | 114.6 rps | 114.2 rps | AIPerf ≥ native |

## Tests

The parity gate runs **in-process** at the library level in `rust/runtime/src/dynosim.rs`
(behind the `dynosim` feature), comparing AIPerf's online driver against Dynamo's own
in-process native driver — no Python subprocess and no external checkout required:

- `online_matches_native_dynamo_live_replay_apples_to_apples` — replays the same
  trace through AIPerf's online driver and Dynamo's public
  `simulate_concurrency_live_requests`, asserting request/token counts exact and
  every latency mean within the 3% bound.
- `online_wall_clock_runs_end_to_end_in_process_without_http` — proves the online
  wall-clock path runs end to end in process with no HTTP/socket.

## Running the parity gate manually

```bash
cargo test --manifest-path rust/Cargo.toml -p aiperf-runtime --features dynosim --lib \
  online_matches_native_dynamo_live_replay_apples_to_apples -- --nocapture
```

For a hand comparison against Dynamo's own CLI, the native side is invoked as:

```bash
PYTHONPATH="$NATIVE/components/src:$NATIVE/lib/bindings/python/src" \
  python -m dynamo.replay <trace.jsonl> --replay-mode online \
  --replay-concurrency 16 --num-workers 1 \
  --extra-engine-args '{"block_size":16}' --trace-format mooncake \
  --report-json <out.json>
```
